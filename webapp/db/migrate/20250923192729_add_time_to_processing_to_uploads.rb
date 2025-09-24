class AddTimeToProcessingToUploads < ActiveRecord::Migration[7.1]
  def change
    add_column :uploads, :time_of_processing, :float
  end
end
