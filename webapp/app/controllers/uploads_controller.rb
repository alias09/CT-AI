class UploadsController < ApplicationController
  def index
    # Эта страница просто отображает форму загрузки
  end

  def history
    @uploads = Upload.all.order(created_at: :desc)
    respond_to do |format|
      format.html
      format.xlsx {
        response.headers['Content-Disposition'] = 'attachment; filename="dicom_analysis_report.xlsx"'
      }
    end
  end

  def upload
    uploaded_files = params[:dicom_zips]

    # 1. Проверяем, были ли файлы загружены
    if uploaded_files.blank?
      redirect_to root_path, alert: "Файлы не были выбраны."
      return
    end

    # 2. Обрабатываем каждый файл
    successful_uploads = 0
    error_messages = []

    # Отфильтровываем пустые значения, которые может прислать форма
    uploaded_files.reject!(&:blank?)

    uploaded_files.each do |uploaded_file|
      # Проверяем, что это zip-архив
      unless uploaded_file.content_type == 'application/zip' || uploaded_file.original_filename.ends_with?('.zip')
        error_messages << "Файл '#{uploaded_file.original_filename}' не является .zip архивом."
        next # Переходим к следующему файлу
      end

      # Сохраняем файл и ставим задачу в очередь
      begin
        uploads_dir = Rails.root.join('tmp', 'uploads')
        FileUtils.mkdir_p(uploads_dir)

        upload = Upload.create!(
          original_filename: uploaded_file.original_filename,
          status: 'pending'
        )

        file_path = uploads_dir.join("#{upload.id}_#{uploaded_file.original_filename}")
        
        File.open(file_path, 'wb') do |file|
          file.write(uploaded_file.read)
        end

        relative_path = file_path.relative_path_from(Rails.root).to_s
        DicomProcessingJob.perform_later(upload.id, relative_path, upload.original_filename)
        
        successful_uploads += 1
      rescue => e
        error_messages << "Ошибка при сохранении файла '#{uploaded_file.original_filename}': #{e.message}"
      end
    end

    # 3. Формируем итоговое сообщение
    notice_message = "Успешно загружено #{successful_uploads} файлов. Они поставлены в очередь на обработку."
    flash[:notice] = notice_message if successful_uploads > 0
    flash[:alert] = error_messages.join("\n") if error_messages.any?

    redirect_to history_path
  end
end
