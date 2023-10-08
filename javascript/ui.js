function mudd_inpainting() {
    var res = Array.from(arguments);

    // get the current selected gallery id
    var idx = selected_gallery_index();
    res[2] = idx; // gallery id

    return submit.apply(this, res);
}

function mudd_inpainting_img2img() {
    var res = Array.from(arguments);

    // get current tabId
    var tab = get_img2img_tab_index();
    var tabid = tab[0]; // tab Id

    var tabs = [ "#img2img_img2img_tab", "#img2img_img2img_sketch_tab", "#img2img_inpaint_tab", "#img2img_inpaint_sketch_tab", "#img2img_inpaint_upload_tab" ];
    if (tabid > 4 || tabid < 0)
        tabid = 0;

    // get base64 src
    var src = gradioApp().querySelector(tabs[tabid]).querySelectorAll("img")[0].src;

    res[3] = src; // gr.Image() component
    res[1] = tab[0]; // tab Id

    var idx = selected_gallery_index();
    res[2] = idx; // gallery Id

    return submit_img2img.apply(this, res);
}

function overlay_masks() {
    var res = Array.from(arguments);

    var label = res[3];
    var is_img2img = res[4];

    var selects = res[0];
    var selected = res[1];
    var options = res[2];
    var sync = false;
    if (typeof options == "boolean") {
        sync = options;
    } else {
        if (options.indexOf("sync") != -1) {
            sync = true;
        }
    }

    var selects = selects.join(",");
    if (sync) {
        selected += "," + selects;
    } else {
        selected = selects;
    }
    var parts = selected.split(",");
    // strip possible label prefix. e.g) "A-face 0.92:3" -> 3
    var sels = [];
    for (var i = 0; i < parts.length; i++) {
        var j, tmp;
        if ((j = parts[i].lastIndexOf(":")) != -1 && parts[i].startsWith(label)) {
            tmp = parts[i].substring(j+1);
        } else {
            tmp = parts[i];
        }

        if ((j = tmp.indexOf("-")) != -1) {
            // expand range e.g) 3-5 -> 3,4,5
            var dummy = tmp.split("-");
            if (dummy.length == 2) {
                var start = parseInt(dummy[0]);
                var end = parseInt(dummy[1]);
                if (isNaN(start) || isNaN(end)) {
                    // ignore
                    continue;
                }
                if (start == end) {
                    sels.push(start);
                    continue;
                }
                if (end < start) {
                    var x = end;
                    end = start;
                    start = x;
                }
                var append = [...Array(end - start + 1).keys()];
                for (var k = 0; k < append.length; k++) {
                    sels.push(append[k] + start);
                }
            }
        } else {
            var num = parseInt(tmp);
            if (!isNaN(num) && num > 0) {
                sels.push(num);
            }
        }
    }

    sels = sels.map((v) => v - 1);

    var tabname = is_img2img ? "img2img":"txt2img";
    var masks_id = "#mudd_masks_" + label.toLowerCase() + "_" + tabname;
    var masks_data = gradioApp().querySelector(masks_id + " textarea");
    var masks = null;
    try {
        masks = JSON.parse(masks_data.value);
    } catch(e) {
        console.log(e);
    }

    if (masks) {
        var canvas = make_mask(masks, sels, is_img2img);
        if (canvas) {
            console.log("canvas created");
        }
    }

    return res;
}

function make_mask(masks, selected, is_img2img) {
    var segms = masks.segms;
    var bboxes = masks.bboxes;
    var labels = masks.labels;
    var scores = masks.scores;

    var tabname = is_img2img ? "img2img" : "txt2img";

    var gallery;
    var gallery_id = "mudd_inpainting_image";
    var suffix = "";
    var imgs;
    if (tabname == "img2img") {
        // get the current tabId
        var tab = get_img2img_tab_index();
        var tabid = tab[0]; // tab Id

        var tabs = [ "img2img_image", "img2img_sketch", "img2maskimg", "inpaint_sketch" ];
        if (tabid > 4 || tabid < 0)
            tabid = 0;
        gallery_id = tabs[tabid];

        gallery = gradioApp().querySelector("#" + gallery_id + ' .image-container');
        imgs = gallery.querySelectorAll("img")
    } else {
        //gallery = gradioApp().querySelector('#' + tabname + '_gallery');
        //imgs = gallery.querySelectorAll(".preview > img");
        gallery = gradioApp().querySelector('#' + gallery_id + ' .image-container');
        imgs = gallery.querySelectorAll("img");
    }

    // check wrapper
    var wrap = gradioApp().querySelectorAll("#" + gallery_id + " .mudd_masks_wrapper")[0];
    if (!wrap) {
        wrap = document.createElement("div");
        wrap.className = "mudd_masks_wrapper";
        wrap.style.display = "flex";
        wrap.style.justifyContent = "center";

        // for gr.Image()
        wrap.style.height = "100%";
        wrap.style.width = "100%";
        wrap.style.position = "absolute";
        wrap.style.top = '0px';

        gallery.appendChild(wrap);
    }

    var canvas = wrap.getElementsByTagName("canvas")[0];
    if (!canvas) {
        canvas = document.createElement("canvas");
        canvas.style.position = "absolute";
        canvas.style.top = '0px';
        canvas.style.zIndex = '100';
        canvas.className = "mask-overlay";
        canvas.style.pointerEvents = "none";

        wrap.appendChild(canvas);
    }

    if (imgs.length > 0) {
        // from https://stackoverflow.com/a/47593316/1696120
        function sfc32(a, b, c, d) {
            return function() {
                a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
                var t = (a + b) | 0;
                a = b ^ b >>> 9;
                b = c + (c << 3) | 0;
                c = (c << 21 | c >>> 11);
                d = d + 1 | 0;
                t = t + d | 0;
                c = c + t | 0;
                return (t >>> 0) / 4294967296;
            }
        }

        var random = random = sfc32(0x9E3779B9, 0x243F6A88, 0xB7E15162, 1377);

        function srand(seed) {
            var seed = seed ^ 0xDEADBEEF; // 32-bit seed with optional XOR value
            // Pad seed with Phi, Pi and E.
            // https://en.wikipedia.org/wiki/Nothing-up-my-sleeve_number
            random = sfc32(0x9E3779B9, 0x243F6A88, 0xB7E15162, seed);
        }

        function randint(min, max) {
            return Math.floor(random() * (max - min + 1)) + min;
        }

        var img = imgs[0];

        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.style.height = img.style.height || "100%";

        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.font = "25px sans-serif";
        for (var i = 0; i < bboxes.length; i++) {
            // always generate colors to fix the order of colors.
            var red = randint(100, 255);
            var green = randint(100, 255);
            var blue = randint(100, 255);

            if (selected.indexOf(i) == -1) {
                continue;
            }

            ctx.fillStyle = 'rgba(' + red + ',' + green + ',' + blue + ',0.3)';
            var label = labels[i];
            var score = scores[i];
            if (score > 0) {
                label += " " + score.toFixed(2);
            }
            label += ":" + (i+1);
            var sz = ctx.measureText(label);
            var bbox = bboxes[i];
            ctx.fillRect(bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]);

            ctx.fillStyle = 'rgba(' + red + ',' + green + ',' + blue + ',0.6)';
            var poly = segms[i];
            for (var k = 0; k < poly.length; k++) {
                var seg = poly[k];
                if (typeof seg[0] == 'number') {
                    // 1-dim array to 2dim array: x,y,x1,y1,x2,y2 -> [x,y], [x1,y1], [x2,y2], ...
                    var tmp = [];
                    while(seg.length)
                        tmp.push(seg.splice(0,2));
                    seg = tmp;
                }

                ctx.beginPath();
                ctx.moveTo(seg[0][0], seg[0][1]);
                for (var j = 1; j < seg.length; j++) {
                    ctx.lineTo(seg[j][0], seg[j][1]);
                }
                ctx.closePath();
                ctx.fill();
            }

            ctx.fillStyle = "#ffffff";
            ctx.fillText(label, (bbox[0] + bbox[2])/2 - sz.width/2, (bbox[1] + bbox[3])/2);
        }

        return canvas;
    }
    return null;
}
