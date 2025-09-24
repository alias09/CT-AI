# frozen_string_literal: true

# URL для подключения к Redis
# redis://<host>:<port>/<db>
redis_url = ENV.fetch('REDIS_URL', 'redis://redis:6379/1')

Sidekiq.configure_server do |config|
  config.redis = { url: redis_url }
end

Sidekiq.configure_client do |config|
  config.redis = { url: redis_url }
end
