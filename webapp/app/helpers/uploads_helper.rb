module UploadsHelper
  def success?(upload)
    upload.status == 'Success'
  end

  def pathology_badge(pathology_status)
    case pathology_status
    when 1
      content_tag(:span, 'Да', class: 'badge bg-danger')
    when 0
      content_tag(:span, 'Нет', class: 'badge bg-success')
    else
      content_tag(:span, 'N/A', class: 'badge bg-secondary')
    end
  end
end
