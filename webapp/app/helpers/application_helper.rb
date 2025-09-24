module ApplicationHelper
  def status_badge_class(status)
    case status
    when 'Success'
      'success'
    when 'processing'
      'info'
    when 'pending'
      'secondary'
    when 'Error'
      'danger'
    else
      'dark'
    end
  end
end
