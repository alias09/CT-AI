require 'zip'
require 'onnxruntime'
require 'caxlsx'

class DicomProcessor
  MODEL_PATH = Rails.root.join('vendor', 'assets', 'ml_models', 'model.onnx').to_s
  IMG_SIZE = 224
  # Стандартные значения нормализации ImageNet
  MEAN = [0.485, 0.456, 0.406]
  STD = [0.229, 0.224, 0.225]

  def self.process(relative_zip_path)
    zip_path = Rails.root.join(relative_zip_path)
    Rails.logger.info "[DicomProcessor] ==> Starting PNG conversion for zip: #{zip_path}"

    study_uid = nil
    series_uid = nil

    # Создаем уникальную директорию для сохранения PNG
    output_dir_name = "dicom_#{Time.now.strftime('%Y%m%d-%H%M%S')}_#{SecureRandom.hex(4)}"
    output_dir_path = Rails.root.join('public', 'converted_pngs', output_dir_name)
    FileUtils.mkdir_p(output_dir_path)

    converted_files = []

    temp_extraction_dir = File.join('tmp', 'dicom_extractions', SecureRandom.uuid)
    FileUtils.mkdir_p(temp_extraction_dir)

    dcm_files_to_process = []

    Zip::File.open(zip_path) do |zip_file|
      # 1. Итерируемся по всем файлам, извлекаем и проверяем, являются ли они DICOM
      zip_file.each do |entry|
        next if entry.name.start_with?('__MACOSX') || entry.directory?

        temp_dcm_path = File.join(temp_extraction_dir, File.basename(entry.name))
        entry.extract(temp_dcm_path) { true } # Извлекаем файл

        # Проверяем, является ли файл действительным DICOM
        if is_dicom_file?(temp_dcm_path)
          dcm_files_to_process << { path: temp_dcm_path, original_name: entry.name }
        else
          Rails.logger.info "[DicomProcessor] Skipping non-DICOM file: #{entry.name}"
          FileUtils.rm_f(temp_dcm_path) # Удаляем ненужный файл
        end
      end
    end

    if dcm_files_to_process.empty?
      FileUtils.rm_rf(temp_extraction_dir)
      return { status: 'Error', message: 'No valid DICOM files found in archive' }
    end

    Rails.logger.info "[DicomProcessor] Found #{dcm_files_to_process.count} valid DICOM files. Starting conversion..."

    begin
      dcm_files_to_process.each_with_index do |file_info, index|
        temp_dcm_path = nil
        begin
          Rails.logger.info "[DicomProcessor] [#{index + 1}/#{dcm_files_to_process.count}] START processing file: #{file_info[:original_name]}"


          temp_dcm_path = file_info[:path]

          # 1. Читаем метаданные для извлечения UID и проверки
          Rails.logger.info "[DicomProcessor] Reading DICOM metadata..."
          meta_dcm = DICOM::DObject.read(temp_dcm_path, max_pixels: 0)
          Rails.logger.info "[DicomProcessor] Metadata read successfully."

          # 2. Проверяем, существует ли тег пиксельных данных. Это дополнительная гарантия.
          unless meta_dcm.exists?('7FE0,0010')
            Rails.logger.warn "[DicomProcessor] Skipping non-image DICOM file (no pixel data): #{file_info[:original_name]}"
            next
          end

          # 3. Извлекаем UID, если они еще не были извлечены
          if study_uid.nil?
            study_uid = meta_dcm.value('0020,000D') # Study Instance UID
            series_uid = meta_dcm.value('0020,000E') # Series Instance UID
            Rails.logger.info "[DicomProcessor] Extracted Study UID: #{study_uid}, Series UID: #{series_uid}"
          end
          
          png_filename = "#{File.basename(file_info[:original_name], '.*')}.png"
          output_png_path = File.join(output_dir_path, png_filename)

          # Передаем пути в наш метод конвертации
          convert_dcm_to_png(temp_dcm_path, output_png_path)
          converted_files << output_png_path

          Rails.logger.info "[DicomProcessor] [#{index + 1}/#{dcm_files_to_process.count}] DONE processing file: #{file_info[:original_name]}"

        rescue => e
          Rails.logger.error "[DicomProcessor] !!! Error processing slice #{file_info[:original_name]}: #{e.message}"
        end
      end
    end

    Rails.logger.info "[DicomProcessor] Conversion finished. #{converted_files.count} files created."

    # ==================================================================
    # Этап 2: Запуск инференса на Python-скрипте
    # ==================================================================
    Rails.logger.info "[DicomProcessor] ==> Starting ML inference..."
    
    python_executable = Rails.root.join('lib', 'python', 'venv', 'bin', 'python')
    script_path = Rails.root.join('lib', 'python', 'run_inference.py')
    model_path = MODEL_PATH # Используем константу класса
    
    command = "#{python_executable} #{script_path} #{output_dir_path} #{model_path}"

    Rails.logger.info "[DicomProcessor] Executing inference command: #{command}"
    stdout, stderr, status = Open3.capture3(command)
    Rails.logger.info "[DicomProcessor] Inference command finished."

    inference_result = {}
    if status.success?
      begin
        parsed_output = JSON.parse(stdout)
        if parsed_output['status'] == 'success'
          inference_result = {
            inference_status: 'completed',
            pathology_detected: parsed_output['pathology_detected'],
            max_probability: parsed_output['max_probability'],
            slices_data: parsed_output['slices_data']
          }
          Rails.logger.info "[DicomProcessor] Inference successful: #{inference_result.inspect}"
        else
          raise "Inference script returned an error: #{parsed_output['message']}"
        end
      rescue JSON::ParserError => e
        raise "Failed to parse inference script output: #{e.message}"
      end
    else
      Rails.logger.error "[DicomProcessor] !!! Inference script failed. Status: #{status.exitstatus}"
      Rails.logger.error "[DicomProcessor] Stderr: #{stderr}"
      raise "Python inference script failed: #{stderr}"
    end

    {
      status: 'Success',
      output_path: output_dir_path.to_s,
      files_converted: converted_files.count,
      study_uid: study_uid,
      series_uid: series_uid
    }.merge(inference_result)


  rescue => e
    Rails.logger.error "[DicomProcessor] !!! An error occurred: #{e.message}"
    Rails.logger.error "[DicomProcessor] Backtrace: \n#{e.backtrace.join("\n")}"
    { status: 'Error', message: e.message }
  ensure
    # Очищаем временную директорию, где были все извлеченные файлы
    Rails.logger.info "[DicomProcessor] Cleaning up temporary directory: #{temp_extraction_dir}"
    FileUtils.rm_rf(temp_extraction_dir) if temp_extraction_dir
  end

  private

  # Проверяет, является ли файл действительным DICOM-файлом, ища "магическое число" 'DICM'
  # по смещению 128 байт.
  def self.is_dicom_file?(file_path)
    return false unless File.exist?(file_path) && File.size(file_path) > 132

    begin
      File.open(file_path, 'rb') do |file|
        file.seek(128)
        magic = file.read(4)
        return magic == 'DICM'
      end
    rescue
      return false
    end
  end

  def self.convert_dcm_to_png(dcm_path, output_path)
    Rails.logger.info "[DicomProcessor] Converting with dcmj2pnm: #{dcm_path} -> #{output_path}"
    
    # Используем современную утилиту dcm2img
    # +Ww <center> <width> : window center and width
    # Формат вывода (PNG) определяется расширением файла
    command = "dcm2img +Ww 40 400 \"#{dcm_path}\" \"#{output_path}\""
    Rails.logger.info "[DicomProcessor] Converting DICOM to PNG with command: #{command}"
    
    success = system(command)
    
    unless success
      raise "dcmj2pnm failed for #{dcm_path}. Check logs for details."
    end

    Rails.logger.debug "[DicomProcessor] dcmj2pnm command executed successfully."
    output_path
  end


end
