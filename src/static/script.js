$(document).ready(function() {
	
	var $anwerContainer = $('#answer-box');

	$('form').on('submit', function(event) {

		$ajax({
			data: {
				question: $('#question-box').val(),
				filename: $('#image-input').val()
			},
			type: 'POST',
			url: '/upload'
			success: function(data) {
				var content = '<figure ><br><br><img id="image" src="/static/' +data["filename"]'" class = "center img-responsive"><br><br><figcaption id = "answer" class="text-center">Mandela</figcaption><br><br></figure>';
			}
		})

		//.done(function(data) {
			//$('#answer-box').show();
			//$('#image').attr('src', '/static/'+data['filename']);

		//});

		event.preventDefault();
	});

});