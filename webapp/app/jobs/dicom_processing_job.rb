class DicomProcessingJob < ApplicationJob
  queue_as :default

  def perform(upload_id, file_path, original_filename)
    Rails.logger.info "[DicomProcessingJob] ==> Received job for Upload ##{upload_id}"
    Rails.logger.info "[DicomProcessingJob] File path: #{file_path}"
    Rails.logger.info "[DicomProcessingJob] Original filename: #{original_filename}"

    upload = Upload.find(upload_id)
    upload.update(status: 'processing')
    Rails.logger.info "[DicomProcessingJob] Updated Upload ##{upload_id} status to 'processing'"

    start_time = Time.now
    result = DicomProcessor.process(file_path)
    processing_time = (Time.now - start_time).round(2)

    update_params = {}
    if result[:status] == 'Success'
      update_params = {
        status: 'completed',
        result_path: result[:output_path]&.sub(Rails.root.join('public').to_s, ''), # Сохраняем относительный путь
        slices_processed: result[:files_converted],
        study_uid: result[:study_uid],
        series_uid: result[:series_uid],
        time_of_processing: processing_time,
        error_message: nil
      }
      
      # Если инференс был успешен, добавляем его результаты
      if result[:inference_status] == 'completed'
        update_params[:probability_of_pathology] = result[:max_probability] # Обновленный ключ
        update_params[:pathology] = result[:pathology_detected] ? 1 : 0
        update_params[:slices_data] = result[:slices_data] # Сохраняем детальные данные
      end
    else
      update_params = {
        status: 'failed',
        error_message: result[:message]
      }
    end

    Rails.logger.info "[DicomProcessingJob] Updating Upload ##{upload_id} with final data: #{update_params.inspect}"
    upload.update(update_params)

    Rails.logger.info "[DicomProcessingJob] <== Finished job for Upload ##{upload_id}"

    # TODO: Удалить исходный zip-файл после обработки
    # File.delete(file_path) if File.exist?(file_path)
  end
end
