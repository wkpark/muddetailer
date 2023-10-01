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
