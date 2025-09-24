FROM ruby:3.3

# Устанавливаем системные зависимости
# Устанавливаем системные зависимости, включая все возможные делегаты для ImageMagick
RUN apt-get update -qq && apt-get install -y \
    build-essential \
    libpq-dev \
    postgresql-client \
    libvips-dev \
    imagemagick \
    libopenjp2-7-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    libtiff-dev \
    libcharls-dev \
    dcmtk \
    libyaml-dev \
    python3 \
    python3-venv \
    curl \
    gpg \
    # Очищаем кэш после установки
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Добавляем репозиторий Yarn (современный способ)
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | gpg --dearmor -o /usr/share/keyrings/yarn-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/yarn-archive-keyring.gpg] https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list

# Устанавливаем yarn
RUN apt-get update && apt-get install -y yarn

# Устанавливаем рабочую директорию
WORKDIR /webapp

# Копируем Gemfile, чтобы установить зависимости
COPY Gemfile Gemfile.lock ./

# Устанавливаем гемы
RUN bundle install

# ==================================================================
# Установка зависимостей Python
# ==================================================================
# Копируем файл с зависимостями
COPY lib/python/requirements.txt ./lib/python/requirements.txt

# Создаем виртуальное окружение и устанавливаем зависимости
RUN python3 -m venv /webapp/lib/python/venv
RUN /webapp/lib/python/venv/bin/pip install --no-cache-dir -r /webapp/lib/python/requirements.txt

# Копируем остальное приложение
COPY . .

# Запускаем сервер
CMD ["bundle", "exec", "rails", "server", "-b", "0.0.0.0"]
