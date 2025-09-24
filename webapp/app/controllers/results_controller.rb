class ResultsController < ApplicationController
  def show
    @upload = Upload.find(params[:id])

    # Проверяем, что у загрузки есть детальные данные по срезам
    unless @upload.slices_data.present?
      redirect_to history_path, alert: 'Детальные результаты для данного исследования не найдены.'
      return
    end

    # Сортируем срезы по вероятности от большей к меньшей
    @sorted_slices = @upload.slices_data.sort_by { |slice| -slice['probability'] }

    # Путь к изображениям уже хранится в нужном формате
    @base_image_path = @upload.result_path
  end
end
