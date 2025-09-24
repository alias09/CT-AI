class AddSlicesDataToUploads < ActiveRecord::Migration[7.1]
  def change
    add_column :uploads, :slices_data, :jsonb
  end
end
