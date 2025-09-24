class AddResultPathToUploads < ActiveRecord::Migration[7.1]
  def change
    add_column :uploads, :result_path, :string
  end
end
