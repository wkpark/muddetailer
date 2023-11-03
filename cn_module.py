"""
muddetailer controlnet support
"""
import gradio as gr
import importlib
import os
import sys

from modules import extensions
from modules.ui import create_refresh_button


cn_extension = None
external_code = None


def init_cn_module():
    global cn_extension, external_code

    if cn_extension is not None and external_code is not None:
        return

    for ext in extensions.active():
        if "controlnet" in ext.name:
            check = os.path.join(ext.path, "scripts", "external_code.py")
            if os.path.exists(check):
                cn_extension = ext
                external_code = importlib.import_module(f"extensions.{ext.name}.scripts.external_code", "external_code")
                print(f" - ControlNet extension {ext.name} found")
                break


def get_cn_models(update=False, types="inpaint,canny,depth,openpose,lineart,softedge,scribble,tile"):
    global external_code, cn_extension

    if cn_extension is None:
        return ["None"]

    if type(types) is str:
        types = [a.strip() for a in types.split(",")]

    models = external_code.get_models(update)
    selected = ["None"] + [model for model in models if any(t in model for t in types)]
    return selected


def get_cn_modules(types="inpaint,canny,depth,openpose,lineart,softedge,scribble,tile"):
    global external_code, cn_extension

    if cn_extension is None:
        return ["None"]

    if type(types) is str:
        types = [a.strip() for a in types.split(",")]

    #modules = external_code.get_modules()
    aliases = external_code.get_modules(True)
    # gradio 4.0.x support tuple choices
    #selected = [("None", "none")] + [(aliases[j], mod) for j, mod in enumerate(modules) if any(t in mod for t in ["inpaint", "tile", "lineart", "openpose", "scribble"])]
    selected = ["None"] + [alias for j, alias in enumerate(aliases) if any(t in alias for t in types)]
    return selected


def get_cn_controls(states):
    global external_code

    cn_states = states.get("controlnet", {})

    model = cn_states.get("model", "None")
    module = cn_states.get("module", "None")

    if model == "None":
        return None

    if module == "None":
        types = [t.strip() for t in "inpaint,canny,depth,openpose,lineart,softedge,scribble,tile".split(",")]
        if any(t in model for t in types):
            for t in types:
                if t in model:
                    # auto detect module
                    modules = get_cn_modules(t)
                    module = modules[1]
                    break
        else:
            return None

    # replace module alias to module
    aliases = external_code.get_modules(True)
    modules = external_code.get_modules()
    if module in modules:
        pass
    elif module in aliases:
        j = aliases.index(module)
        module = modules[j]
    else:
        # not found?
        return None

    control_mode = cn_states.get("control_model", external_code.ControlMode.BALANCED)
    weight = cn_states.get("weight", 1)
    guidance_start = cn_states.get("guidance_start", 0)
    guidance_end = cn_states.get("guidance_end", 1)
    pixel_perfect = cn_states.get("pixel_perfect", True)

    return [model, module, weight, guidance_start, guidance_end, control_mode, pixel_perfect]


def cn_unit(p, model, module, weight=1, guidance_start=0, guidance_end=1, control_mode=None, pixel_perfect=True):
    global external_code, cn_extension

    if cn_extension is None:
        return None

    control_mode = external_code.ControlMode.BALANCED if control_mode is None else control_mode
    return external_code.ControlNetUnit(
        model=model,
        module=module,
        weight=weight,
        control_mode=control_mode,
        guidance_start=guidance_start,
        guidance_end=guidance_end,
        pixel_perfect=pixel_perfect,
    )


def cn_control_ui(is_img2img=False):
    global external_code

    with gr.Column(variant="compact"):
        with gr.Row():
            pixel_perfect = gr.Checkbox(
                label="Pixel Perfect",
                value=True,
                #elem_id=f"{elem_id_tabname}_{tabname}_controlnet_pixel_perfect_checkbox",
            )

        with gr.Row(elem_classes=["controlnet_preprocessor_model", "controlnet_row"]):
            module = gr.Dropdown(
                choices=get_cn_modules(),
                label=f"Preprocessor",
                value="None",
                #elem_id=f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_dropdown",
            )
            model = gr.Dropdown(
                get_cn_models(),
                label=f"Model",
                value="None",
                #elem_id=f"{elem_id_tabname}_{tabname}_controlnet_model_dropdown",
            )
            create_refresh_button(model, lambda: None , lambda: {"choices": get_cn_models(True)}, "mudd_refresh_cn_models")


            types = [t.strip() for t in "inpaint,canny,depth,openpose,lineart,softedge,scribble,tile".split(",")]
            def match_model_module(model, module):
                """match model and module"""
                if any(t in model for t in types):
                    for t in types:
                        if t in model:
                            choices = get_cn_modules(t)
                            if module == "None" or module not in choices:
                                module = choices[1]
                                return gr.update(choices=choices, value=module)
                            return gr.update(choices=choices)

                choices = get_cn_modules()
                return gr.update(choices=choices)


            def match_module_model(model, module):
                """match model and module"""
                if any(t in module for t in types):
                    for t in types:
                        if t in module:
                            choices = get_cn_models(False, t)
                            if model == "None" or model not in choices:
                                model = choices[1]
                                return gr.update(value=model)
                            return gr.update()

                return gr.update()


            model.select(
                fn=match_model_module,
                inputs=[model, module],
                outputs=[module],
            )

            module.select(
                fn=match_module_model,
                inputs=[model, module],
                outputs=[model],
            )


        with gr.Row(elem_classes=["controlnet_weight_steps", "controlnet_row"]):
            weight = gr.Slider(
                label=f"Control Weight",
                value=1,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                #elem_id=f"{elem_id_tabname}_{tabname}_controlnet_control_weight_slider",
                elem_classes="controlnet_control_weight_slider",
            )
            guidance_start = gr.Slider(
                label="Starting Control Step",
                value=0,
                minimum=0.0,
                maximum=1.0,
                interactive=True,
                #elem_id=f"{elem_id_tabname}_{tabname}_controlnet_start_control_step_slider",
                elem_classes="controlnet_start_control_step_slider",
            )
            guidance_end = gr.Slider(
                label="Ending Control Step",
                value=1,
                minimum=0.0,
                maximum=1.0,
                interactive=True,
                #elem_id=f"{elem_id_tabname}_{tabname}_controlnet_ending_control_step_slider",
                elem_classes="controlnet_ending_control_step_slider",
            )
    control_mode = gr.Radio(
        choices=[e.value for e in external_code.ControlMode],
        value=external_code.ControlMode.BALANCED.value,
        label="Control Mode",
        #elem_id=f"{elem_id_tabname}_{tabname}_controlnet_control_mode_radio",
        elem_classes="controlnet_control_mode_radio",
    )

    return model, module, weight, guidance_start, guidance_end, control_mode, pixel_perfect


def _disable_controlnet_units(p):
    global external_code

    units = external_code.get_all_units_in_processing(p)
    for unit in units:
        if hasattr(unit, "enabled"):
            unit.enabled = False


def update_cn_script_in_processing(p, units):
    global external_code, cn_extension

    if cn_extension is None:
        return None

    # disable all units of controlnet
    _disable_controlnet_units(p)

    # use uddetailer's cn_units
    external_code.update_cn_script_in_processing(p, units)
