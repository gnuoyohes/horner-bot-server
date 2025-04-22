const CLASSES = {
    0: 'Person',
    15: 'Cat',
    67: 'CellPhone'
}

// var socket = io.connect('10.0.0.39:5000');
var socket = io({transports: ['websocket']})

socket.on('connect', function() {
    console.log(socket.io.engine.transport.name);
})


// socket.on('video_frame', async data => {
//     $('#videoStream').attr('src', 'data:image/jpeg;base64,' + data.image);
// })

var clientFps = 0
var imageURL = null

async function fetchFrame() {
    // startTime = performance.now();
    fetch('/video_frame').then(response => {
        return response.blob()
    })
    .then(blob => {
        if (imageURL) {
            URL.revokeObjectURL(imageURL)
        }
        imageURL = URL.createObjectURL(blob);
        $('#videoStream').attr('src', imageURL);
        
        // endTime = performance.now();
        // clientFps = Math.round(1000 / (endTime - startTime) * 10) / 10
    })
    
    // $('#videoStream').attr('src', 'data:image/jpeg;base64,' + response.body);
    // $('#videoStream').css('background-image', 'url(' + imageObjectURL + ')');
}

socket.on('state', function(data) {
    console.log(socket.io.engine.transport.name);
    for (const key in data) {
        switch (key) {
            case 'use_yolo':
                $('#yoloSwitch').prop('checked', data[key])
                if (data[key])
                    $('#yoloCheckboxes').show()
                else
                    $('#yoloCheckboxes').hide()
                break
            case 'laser_running':
                var laserBtn = $('#btn_start_laser')
                var laserPoint = $('#laser-point')
                if (data[key]) {
                    laserBtn.addClass('btn-danger')
                    laserBtn.removeClass('btn-primary')
                    laserBtn.text('Stop Laser')
                    laserPoint.css("background-color", "#dc3545");
                }
                else {
                    laserBtn.addClass('btn-primary')
                    laserBtn.removeClass('btn-danger')
                    laserBtn.text('Start Laser')
                    laserPoint.css("background-color", "#0d6efd");
                }
                break
            case 'minimum_contour_area':
                var slider = $('#minContourAreaSlider')
                var sliderValLabel = $('#minContourAreaVal')
                slider.val(data[key])
                sliderValLabel.text(data[key])
                break
            case 'manual_mode':
                $('#manualModeSwitch').prop('checked', data[key])
                var laserPoint = $('#laser-point')
                if (data[key]) {
                    laserPoint.removeClass('blocked')
                    laserPoint.css('border-style', 'dotted')
                }
                else {
                    laserPoint.addClass('blocked')
                    laserPoint.css('border-style', 'none')
                }
                break
            case 'classes_to_detect':
                for (const class_id in CLASSES) {
                    const isChecked = data[key].includes(parseInt(class_id))
                    $(`#detect${CLASSES[class_id]}Radio`).prop('checked', isChecked)
                }
        }
    }
})

var now = Date.now()

socket.on('stats', function(stats) {
    const newNow = Date.now()
    if (Date.now() - now > 500) {
        // $('#client-fps').text(`Client FPS: ${clientFps}`)
        $('#server-fps').text(`Server FPS: ${stats['server_fps']}`)
        $('#cpu').text(`CPU Usage: ${stats['cpu_percent']}`)
        $('#ram').text(`Memory Usage: ${stats['memory_percent']}`)
        now = newNow
    }
})

socket.on('laser_coords', function(coords) {
    if (!dragging) {
        var coordsX = coords[0]
        var coordsY = coords[1]
        laserPositionScaled = [coordsX, coordsY]
        updateLaserPosition(coordsX, coordsY)
    }
})

function updateState(params) {
    socket.emit('update_state', params)
}

function updateLaserCoords(coords) {
    socket.emit('update_laser_coords', coords)
}

function handleYoloCheckbox(el) {
    updateState({'use_yolo': el.checked})
}

function handleYoloDetectCheckbox(el) {
    for (const class_id in CLASSES) {
        if (el.id == `detect${CLASSES[class_id]}Radio`) {
            socket.emit('update_class', [class_id, el.checked])
            break
        }
    }
}

function handleManualModeCheckbox(el) {
    updateState({'manual_mode': el.checked})
}

function handleLaser(el) {
    const isRunning = el.innerHTML == 'Start Laser';
    updateState({'laser_running': isRunning})
}

function handleMinContourAreaSlider(el) {
    const sliderVal = el.value
    console.log(sliderVal)
    updateState({'minimum_contour_area': sliderVal})
}

var laserPosition = { x: 0, y: 0 }
var laserPositionScaled = [0, 0]
const laserWidth = $('#laser-point').outerWidth()
const laserHeight = $('#laser-point').outerHeight()
var containerWidth = $('#laser-container').outerWidth()
var containerHeight = $('#laser-container').outerHeight()
var dragging = false

interact('.draggable:not(.blocked)').draggable({
    listeners: {
        start (event) {
            console.log(event.type, event.target)
            dragging = true
        },
        move (event) {
            laserPosition.x += event.dx
            laserPosition.y += event.dy
            
            let posScaled = [
                (laserPosition.x + 1) / (containerWidth - laserWidth),
                (laserPosition.y + 1) / (containerHeight - laserHeight)
            ]
            // console.log(posScaled)
            laserPositionScaled = posScaled
            updateLaserCoords(posScaled)
            $('#laser-point').css('transform', `translate(${laserPosition.x}px, ${laserPosition.y}px)`)
        },
        end (event) {
            dragging = false
        }
    },
    modifiers: [
        interact.modifiers.restrictRect({
            restriction: 'parent',
        })
    ]
})

function updateLaserPosition(newXScaled, newYScaled) {
    laserPosition.x = newXScaled * (containerWidth - laserWidth) - 1
    laserPosition.y = newYScaled * (containerHeight - laserHeight) - 1
    $('#laser-point').css('transform', `translate(${laserPosition.x}px, ${laserPosition.y}px)`)
}

window.addEventListener('resize', function() {
    containerWidth = $('#laser-container').outerWidth()
    containerHeight = $('#laser-container').outerHeight()
    updateLaserPosition(laserPositionScaled[0], laserPositionScaled[1])
});

setInterval(fetchFrame, 16)