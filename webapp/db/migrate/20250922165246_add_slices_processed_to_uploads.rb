class AddSlicesProcessedToUploads < ActiveRecord::Migration[7.1]
  def change
    add_column :uploads, :slices_processed, :integer
  end
end
