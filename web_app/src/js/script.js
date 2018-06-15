$(document).ready(function() {
	
	

	$('form').on('submit', function(event) {

		$ajax({
			data: {
				question: $('#question-box').val(),
				filename: $('#image-input').val()
			},
			type: 'POST',
			url: '/'
		}).done(function(data) {
			$('#answer-box').show();
			$('#image').attr('src', '/static/'+data['filename']);

		});

		event.preventDefault();
	});

});