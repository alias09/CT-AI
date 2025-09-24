class AddErrorMessageToUploads < ActiveRecord::Migration[7.1]
  def change
    add_column :uploads, :error_message, :text
  end
end
