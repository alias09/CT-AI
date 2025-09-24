class CreateUploads < ActiveRecord::Migration[7.1]
  def change
    create_table :uploads do |t|
      t.string :original_filename
      t.string :report_path
      t.string :status
      t.string :study_uid
      t.string :series_uid
      t.float :probability_of_pathology
      t.integer :pathology

      t.timestamps
    end
  end
end
