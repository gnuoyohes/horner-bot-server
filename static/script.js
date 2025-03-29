// var socket = io.connect('10.0.0.39:5000');
var socket = io();

socket.on('video_frame', function(data) {
    var videoElement = document.getElementById('video');
    videoElement.src = 'data:image/jpeg;base64,' + data.image;
});

function switchObjectDetection() {
    socket.emit('switch_object_detection')
}