
/* global $ */
class Main {
    constructor() {
        this.canvas = document.getElementById('main');
        this.input = document.getElementById('input');
        this.canvas.width  = 449; // 16 * 28 + 1
        this.canvas.height = 449; // 16 * 28 + 1
        this.ctx = this.canvas.getContext('2d');
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mouseup',   this.onMouseUp.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.initialize();
    }
    initialize() {
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(0, 0, 449, 449);
        // this.ctx.lineWidth = 1;
        // this.ctx.strokeRect(0, 0, 449, 449);
        // this.ctx.lineWidth = 0.05;
        // for (var i = 0; i < 27; i++) {
        //     this.ctx.beginPath();
        //     this.ctx.moveTo((i + 1) * 16,   0);
        //     this.ctx.lineTo((i + 1) * 16, 449);
        //     this.ctx.closePath();
        //     this.ctx.stroke();
        //
        //     this.ctx.beginPath();
        //     this.ctx.moveTo(  0, (i + 1) * 16);
        //     this.ctx.lineTo(449, (i + 1) * 16);
        //     this.ctx.closePath();
        //     this.ctx.stroke();
        // }
        this.drawInput();
        $('#output td').text('').removeClass('success');
    }
    onMouseDown(e) {
        this.canvas.style.cursor = 'default';
        this.drawing = true;
        this.prev = this.getPosition(e.clientX, e.clientY);
    }
    onMouseUp() {
        this.drawing = false;
        this.drawInput();
    }
    onMouseMove(e) {
        if (this.drawing) {
            var curr = this.getPosition(e.clientX, e.clientY);
            this.ctx.lineWidth = 25;
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();
            this.ctx.moveTo(this.prev.x, this.prev.y);
            this.ctx.lineTo(curr.x, curr.y);
            this.ctx.stroke();
            this.ctx.closePath();
            this.prev = curr;
        }
    }
    getPosition(clientX, clientY) {
        var rect = this.canvas.getBoundingClientRect();
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }
    drawInput() {
        var ctx = this.input.getContext('2d');
        var img = new Image();
        img.onload = () => {
            var small = document.createElement('canvas').getContext('2d');
            small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 64, 64);
            var data = small.getImageData(0, 0, 64, 64).data;
            for (var i = 0; i < 64; i++) {
                for (var j = 0; j < 64; j++) {
                    var n = 4 * (i * 64 + j);
                    ctx.fillStyle = 'rgb(' + [data[n + 0], data[n + 1], data[n + 2]].join(',') + ')';
                    ctx.fillRect(j * 1, i * 1, 1, 1);
                }
            }
        };
        img.src = this.canvas.toDataURL();
    }


      fileUPInput() {
        var dataurl = this.input.toDataURL("image/png");
        var arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1],
            bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
        while (n--) {
            u8arr[n] = bstr.charCodeAt(n);
        }
        var blob = new Blob([u8arr], {type: mime});

        var fd = new FormData();
        fd.append("image", blob, "image.png");

        var datas = fd;
        // alert(inputs)
        $.ajax({
            url: '/api/mnist',
            method: 'Post',
            type: 'POST',
            processData: false,
            contentType: false,
            // contentType: 'application/json',
            data: datas,
            success: (data) => {
                for (let i = 0; i < 10; i++)
                {
                    $('#output tr').eq(i + 1).find('td').eq(0).text(data.results[0][i]);
                    $('#output tr').eq(i + 1).find('td').eq(1).text(data.results[1][i]);
                }
                // alert(data)
                // for (let i = 0; i < 2; i++) {
                //     var max = 0;
                //     var max_index = 0;
                //     for (let j = 0; j < 10; j++) {
                //         var value =data.results[i][j] ;
                //         // if (value > max) {
                //         //     max = value;
                //         //     max_index = j;
                //         // }
                //         // var digits = String(value).length;
                //         // for (var k = 0; k < 3 - digits; k++) {
                //         //     value = '0' + value;
                //         // }
                //         // var text = '0.' + value;
                //         // if (value > 999) {
                //         //     text = '1.000';
                //         // }
                //         $('#output tr').eq(j + 1).find('td').eq(i).text(text);
                //     }
                //     for (let j = 0; j < 10; j++) {
                //         if (j === max_index) {
                //             $('#output tr').eq(j + 1).find('td').eq(i).addClass('success');
                //         } else {
                //             $('#output tr').eq(j + 1).find('td').eq(i).removeClass('success');
                //         }
                //     }
                // }
            }
        });
        // };
        // $('#input_image').attr('src', this.canvas.toDataURL("image/png"));
    }
}

$(() => {
    var main = new Main();
    $('#clear').click(() => {
        main.initialize();
    });
    $('#text').click(() => {
        main.fileUPInput()
    });


});


