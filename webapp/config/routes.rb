Rails.application.routes.draw do
  # Маршрут для просмотра результатов конвертации
  get 'results/:id', to: 'results#show', as: :result
  # Устанавливаем главную страницу
  root "uploads#index"

  # Добавляем маршрут для загрузки и обработки файла
  post "uploads/upload" => "uploads#upload", as: :upload_file

  # Добавляем маршрут для страницы истории
  get "uploads/history" => "uploads#history", as: :history

  # Reveal health status on /up that returns 200 if the app boots with no exceptions, otherwise 500.
  # Can be used by load balancers and uptime monitors to verify that the app is live.
  get "up" => "rails/health#show", as: :rails_health_check
end
