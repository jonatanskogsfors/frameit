import marimo

__generated_with = "0.25.0"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _():
    import base64
    import io
    from dataclasses import dataclass, replace
    from pathlib import Path

    import marimo as mo
    import numpy as np
    from PIL import Image

    from frameit import convert

    return Image, Path, base64, convert, dataclass, io, mo, np, replace


@app.cell(hide_code=True)
def _(convert, dataclass, mo):
    @dataclass(frozen=True)
    class Selection:
        folder: str = "kiki2"
        palette_id: str = "acep"
        preset_id: str = "animated"
        resample: convert.Resample = "lanczos"
        output_format: convert.OutputFormat = "bmp"
        settings_path: str = ""

    PRESETS = convert.load_presets()
    NEUTRAL_TUNING = convert.Tuning(
        flatten_threshold=0.0,
        gamut_strength=0.0,
        hue_pull_strength=0.0,
        dither_strength=1.0,
        chroma_weight=1.0,
    )
    FILE_PALETTES = convert.load_palettes()

    _preset = PRESETS[Selection.preset_id]
    get_tuning, set_tuning = mo.state(_preset.tuning)
    get_selection, set_selection = mo.state(Selection(resample=_preset.resample))
    get_image_index, set_image_index = mo.state(0)
    # A palette from a settings file that matches none of the palette files
    get_loaded_palette, set_loaded_palette = mo.state(None)
    get_load_message, set_load_message = mo.state("")
    return (
        FILE_PALETTES,
        NEUTRAL_TUNING,
        PRESETS,
        get_image_index,
        get_load_message,
        get_loaded_palette,
        get_selection,
        get_tuning,
        set_image_index,
        set_load_message,
        set_loaded_palette,
        set_selection,
        set_tuning,
    )


@app.cell(hide_code=True)
def _(
    NEUTRAL_TUNING,
    PRESETS,
    get_selection,
    mo,
    replace,
    set_selection,
    set_tuning,
):
    # Own cell, so that the slider cell is redrawn when a reset button is clicked
    def _reset_button(label, names, source):
        def _reset(clicks):
            values = {name: getattr(source, name) for name in names}
            set_tuning(lambda tuning: replace(tuning, **values))
            return (clicks or 0) + 1

        return mo.ui.button(label=label, value=0, on_click=_reset)

    reset_tone = _reset_button(
        "↺ Neutral", ["gamma", "contrast", "saturation"], NEUTRAL_TUNING
    )
    reset_gamut = _reset_button(
        "↺ Neutral",
        ["flatten_threshold", "flatten_window", "gamut_strength"],
        NEUTRAL_TUNING,
    )
    reset_hue = _reset_button(
        "↺ Neutral",
        ["hue_pull_strength", "hue_pull_range", "snap_radius"],
        NEUTRAL_TUNING,
    )
    reset_dither = _reset_button(
        "↺ Neutral", ["dither_strength", "chroma_weight"], NEUTRAL_TUNING
    )

    def _reset_to_preset(clicks):
        preset = PRESETS[get_selection().preset_id]
        set_tuning(preset.tuning)
        set_selection(lambda selection: replace(selection, resample=preset.resample))
        return (clicks or 0) + 1

    reset_all = mo.ui.button(
        label="↺ Återställ alla till utgångsläget", value=0, on_click=_reset_to_preset
    )
    return reset_all, reset_dither, reset_gamut, reset_hue, reset_tone


@app.cell(hide_code=True)
def _(FILE_PALETTES, Path, convert, get_loaded_palette, get_selection, get_tuning):
    LOADED_PALETTE_ID = "frameit.json"
    PALETTES = dict(FILE_PALETTES)
    if get_loaded_palette() is not None:
        PALETTES[LOADED_PALETTE_ID] = get_loaded_palette()

    def frame_folder(folder: str) -> Path:
        source = Path(folder.strip())
        return source.parent / (source.name + "_frame")

    def default_settings_path(folder: str) -> Path:
        return frame_folder(folder) / "frameit.json"

    def current_settings() -> convert.Settings:
        selection = get_selection()
        return convert.Settings(
            palette=PALETTES[selection.palette_id],
            resample=selection.resample,
            output_format=selection.output_format,
            tuning=get_tuning(),
            preset=selection.preset_id,
        )

    return (
        LOADED_PALETTE_ID,
        PALETTES,
        current_settings,
        default_settings_path,
        frame_folder,
    )


@app.cell(hide_code=True)
def _(
    FILE_PALETTES,
    LOADED_PALETTE_ID,
    PRESETS,
    Path,
    convert,
    default_settings_path,
    get_selection,
    mo,
    replace,
    set_load_message,
    set_loaded_palette,
    set_selection,
    set_tuning,
):
    # Own cell, so that the slider cell is redrawn with the loaded values
    def _load(clicks):
        selection = get_selection()
        typed_path = selection.settings_path.strip()
        path = Path(typed_path) if typed_path else default_settings_path(selection.folder)
        try:
            settings = convert.Settings.load(path)
        except Exception as error:
            set_load_message(f"Kunde inte läsa {path}: {error}")
            return (clicks or 0) + 1

        content_hash = settings.palette.content_hash
        matching_id = next(
            (
                palette_id
                for palette_id, palette in FILE_PALETTES.items()
                if palette.content_hash == content_hash
            ),
            None,
        )
        if matching_id:
            set_loaded_palette(None)
            set_load_message(
                f"Läste in {path}. Paletten är {matching_id}.json ({content_hash})."
            )
        else:
            set_loaded_palette(settings.palette)
            set_load_message(
                f"Läste in {path}. Paletten i filen ({content_hash}) matchar ingen "
                "palettfil och används som den är."
            )
        set_tuning(settings.tuning)
        set_selection(
            lambda current: replace(
                current,
                palette_id=matching_id or LOADED_PALETTE_ID,
                preset_id=(
                    settings.preset if settings.preset in PRESETS else current.preset_id
                ),
                resample=settings.resample,
                output_format=settings.output_format,
            )
        )
        return (clicks or 0) + 1

    load_button = mo.ui.button(label="Läs in", value=0, on_click=_load)
    return (load_button,)


@app.cell(hide_code=True)
def _(
    LOADED_PALETTE_ID,
    PALETTES,
    PRESETS,
    default_settings_path,
    get_load_message,
    get_selection,
    get_tuning,
    load_button,
    mo,
    replace,
    reset_all,
    reset_dither,
    reset_gamut,
    reset_hue,
    reset_tone,
    set_selection,
    set_tuning,
):
    # The panel is shown by the cell that creates the sliders, so that it keeps its
    # scroll position when a value changes
    _tuning = get_tuning()
    _selection = get_selection()

    def _slider(name, start, stop, step, label):
        def _update(value):
            set_tuning(lambda tuning: replace(tuning, **{name: value}))

        return mo.ui.slider(
            start,
            stop,
            step=step,
            value=getattr(_tuning, name),
            label=label,
            on_change=_update,
            show_value=True,
            debounce=True,
            full_width=True,
        )

    def _select(name):
        def _update(value):
            set_selection(lambda selection: replace(selection, **{name: value}))

        return _update

    def _dropdown(options, name, label):
        current = getattr(_selection, name)
        return mo.ui.dropdown(
            options=options,
            value=next(text for text, value in options.items() if value == current),
            label=label,
            on_change=_select(name),
        )

    def _with_hint(slider, hint):
        return mo.vstack(
            [
                slider,
                mo.Html(
                    '<div style="font-size:0.75em;opacity:0.6;margin:-6px 0 4px 0">'
                    f"{hint}</div>"
                ),
            ],
            gap=0,
        )

    def _heading(title, button):
        return mo.hstack(
            [mo.md(f"#### {title}"), button], justify="space-between", align="center"
        )

    def _small(text):
        return mo.Html(f'<div style="font-size:0.75em;opacity:0.6">{text}</div>')

    _palette_options = {
        f"{palette.name} · {palette.content_hash}"
        + (" (från frameit.json)" if palette_id == LOADED_PALETTE_ID else ""): palette_id
        for palette_id, palette in PALETTES.items()
    }
    folder_input = mo.ui.text(
        value=_selection.folder,
        label="Mapp med bilder",
        full_width=True,
        on_change=_select("folder"),
    )
    palette_dropdown = _dropdown(_palette_options, "palette_id", "Panel")
    resample_dropdown = _dropdown(
        {"Lanczos (film)": "lanczos", "Närmsta granne (pixelgrafik)": "nearest"},
        "resample",
        "Skalning",
    )

    def _choose_preset(preset_id):
        preset = PRESETS[preset_id]
        set_tuning(preset.tuning)
        set_selection(
            lambda selection: replace(
                selection, preset_id=preset_id, resample=preset.resample
            )
        )

    preset_dropdown = mo.ui.dropdown(
        options={preset.name: preset_id for preset_id, preset in PRESETS.items()},
        value=PRESETS[_selection.preset_id].name,
        label="Utgångsläge",
        on_change=_choose_preset,
    )
    settings_path_input = mo.ui.text(
        value=_selection.settings_path,
        label="Parameterfil",
        placeholder=str(default_settings_path(_selection.folder)),
        full_width=True,
        on_change=_select("settings_path"),
    )

    gamma = _slider("gamma", 0.5, 1.5, 0.05, "Gamma")
    contrast = _slider("contrast", 0.5, 2.0, 0.05, "Kontrast")
    saturation = _slider("saturation", 0.0, 2.5, 0.05, "Mättnad")

    flatten_threshold = _slider(
        "flatten_threshold", 0.0, 0.05, 0.002, "Utslätning: tröskel"
    )
    flatten_window = _slider("flatten_window", 3, 15, 2, "Utslätning: fönster (px)")
    gamut_strength = _slider("gamut_strength", 0.0, 1.0, 0.05, "Gamut-mappning")

    hue_pull_strength = _slider("hue_pull_strength", 0.0, 1.0, 0.05, "Nyansdragning")
    hue_pull_range = _slider("hue_pull_range", 5, 60, 5, "Nyansdragning: räckvidd (°)")
    snap_radius = _slider("snap_radius", 0.0, 0.25, 0.01, "Snäppning: radie")

    dither_strength = _slider("dither_strength", 0.0, 1.0, 0.05, "Felspridning")
    chroma_weight = _slider("chroma_weight", 1.0, 5.0, 0.5, "Kulörvikt")

    output_format_dropdown = _dropdown(
        {"BMP 24-bitar (PhotoPainter)": "bmp", "PNG indexerad (Inky Frame)": "png"},
        "output_format",
        "Format",
    )
    export_button = mo.ui.run_button(label="💾 Spara alla bilder")
    save_settings_button = mo.ui.run_button(label="Spara bara parameterfil")

    view_step = mo.ui.radio(
        options=[
            "Original",
            "1. Ton",
            "2. Gamut",
            "3. Nyans/snäpp",
            "4. Förhandsvisning",
            "Panelbild",
        ],
        value="4. Förhandsvisning",
        inline=True,
    )
    compare_original = mo.ui.checkbox(label="Visa originalet under")

    mo.vstack(
        [
            mo.md("#### Källa"),
            folder_input,
            mo.hstack([palette_dropdown, resample_dropdown], justify="start"),
            preset_dropdown,
            _small(PRESETS[_selection.preset_id].description),
            mo.hstack([settings_path_input, load_button], align="end", widths=[1, 0]),
            _small(get_load_message()),
            reset_all,
            _heading("1. Ton", reset_tone),
            _with_hint(
                gamma,
                "← ljusare skuggor, plattare · mörkare skuggor, mer djup →"
                "<br>Neutral: 1,0",
            ),
            _with_hint(
                contrast,
                "← plattare, gråare · hårdare, ljust och mörkt klipps →<br>Neutral: 1,0",
            ),
            _with_hint(saturation, "← mot gråskala · starkare färger →<br>Neutral: 1,0"),
            _heading("2. Utslätning och gamut", reset_gamut),
            _with_hint(
                flatten_threshold,
                "← färre ytor slätas ut · fler, risk att gradienter jämnas ut →"
                "<br>Neutral: 0 (av)",
            ),
            _with_hint(
                flatten_window,
                "← bara smått brus · större ytor, mjukare kanter →"
                "<br>Påverkar bara när tröskeln är över 0",
            ),
            _with_hint(
                gamut_strength,
                "← färger orörda, klipps senare · pressas in i panelens omfång →"
                "<br>Neutral: 0 (av). Rekommenderat: 1,0",
            ),
            _heading("3. Nyans och snäppning", reset_hue),
            _with_hint(
                hue_pull_strength,
                "← trogna nyanser · dras mot pigmenten →<br>Neutral: 0 (av)",
            ),
            _with_hint(
                hue_pull_range,
                "← bara färger nära ett pigment · även färger längre bort →"
                "<br>Påverkar bara när nyansdragningen är över 0",
            ),
            _with_hint(
                snap_radius,
                "← färre enfärgade fält · fler, nyanser kan bli fel →<br>Neutral: 0 (av)",
            ),
            _heading("4. Dithering", reset_dither),
            _with_hint(
                dither_strength,
                "← färre prickar, mer banding · mjukare övergångar →"
                "<br>Neutral: 1,0 (vanlig Floyd–Steinberg)",
            ),
            _with_hint(
                chroma_weight,
                "← kulör och ljushet lika viktiga · neutrala toner får svart/vitt →"
                "<br>Neutral: 1,0",
            ),
            mo.md("#### 5. Spara till SD-kort"),
            mo.hstack(
                [output_format_dropdown, export_button, save_settings_button],
                justify="start",
                align="center",
            ),
            _small(
                "Alla bilder i mappen bearbetas med inställningarna ovan och sparas i en "
                "mapp bredvid källmappen, med “_frame” efter namnet. Befintliga filer "
                "skrivs över.<br>BMP: mappen får samma struktur som SD-kortet (bilder i "
                "pic/, fileList.txt för läge 2). PhotoPainter klarar högst 2000 bilder."
                "<br>Inställningarna och paletten sparas i frameit.json i samma mapp. "
                "“Spara bara parameterfil” sparar enbart den."
            ),
        ],
        gap=0.4,
    ).style({"max-height": "88vh", "overflow-y": "auto", "padding-right": "12px"})
    return (
        chroma_weight,
        compare_original,
        contrast,
        dither_strength,
        export_button,
        flatten_threshold,
        flatten_window,
        folder_input,
        gamma,
        gamut_strength,
        hue_pull_range,
        hue_pull_strength,
        palette_dropdown,
        resample_dropdown,
        saturation,
        save_settings_button,
        snap_radius,
        view_step,
    )


@app.cell(hide_code=True)
def _(
    convert, current_settings, export_button, frame_folder, get_selection, image_files, mo
):
    mo.stop(not export_button.value)

    _settings = current_settings()
    _output_folder = frame_folder(get_selection().folder)
    with mo.status.progress_bar(
        total=len(image_files), title="Sparar bilder", remove_on_exit=True
    ) as _progress:
        _result = convert.export_images(
            image_files, _settings, _output_folder, on_progress=_progress.update
        )

    _message = (
        f"**Sparade {len(_result.file_names)} av {len(image_files)} bilder** "
        f"i `{_output_folder}`"
    )
    if _settings.output_format == "bmp":
        _message += (
            "<br>Kopiera innehållet i mappen till SD-kortets rot "
            "och radera `index.txt` på kortet för att börja från första bilden."
        )
    if _result.errors:
        _message += f"<br>{len(_result.errors)} fel, t.ex. {_result.errors[0]}"
    mo.callout(mo.md(_message), kind="warn" if _result.errors else "success")
    return


@app.cell(hide_code=True)
def _(current_settings, default_settings_path, get_selection, mo, save_settings_button):
    mo.stop(not save_settings_button.value)

    _path = default_settings_path(get_selection().folder)
    _path.parent.mkdir(parents=True, exist_ok=True)
    current_settings().save(_path)
    mo.callout(mo.md(f"**Sparade parameterfilen** `{_path}`"), kind="success")
    return


@app.cell(column=1, hide_code=True)
def _(
    base64,
    compare_original,
    current_file,
    gamut_image,
    image_files,
    image_slider,
    io,
    load_error,
    mo,
    next_button,
    np,
    palette,
    palette_indices,
    panel_image,
    preview_image,
    previous_button,
    pulled_image,
    scaled_image,
    snapped_share,
    toned_image,
    view_step,
):
    def _pixel_exact(image):
        # Browser scaling causes moiré in the dithering, so show it at exactly 1:1
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return mo.Html(
            '<div style="overflow-x:auto">'
            f'<img src="data:image/png;base64,{encoded}" width="{image.width}" '
            'style="image-rendering:pixelated;display:block"></div>'
        )

    _images_by_step = {
        "Original": scaled_image,
        "1. Ton": toned_image,
        "2. Gamut": gamut_image,
        "3. Nyans/snäpp": pulled_image,
        "4. Förhandsvisning": preview_image,
        "Panelbild": panel_image,
    }

    _shares = (
        np.bincount(palette_indices.ravel(), minlength=len(palette.measured_rgb))
        / palette_indices.size
    )
    _swatches = "".join(
        '<span style="display:inline-flex;align-items:center;margin-right:12px">'
        '<span style="width:16px;height:16px;border:1px solid #888;margin-right:4px;'
        f'background:rgb{tuple(color)}"></span>{name} {share:.0%}</span>'
        for color, name, share in zip(
            palette.measured_rgb, palette.color_names, _shares, strict=True
        )
    )

    _position = (
        f"{min(image_slider.value, len(image_files) - 1) + 1} / {len(image_files)} · "
        f"{current_file.name}"
        if current_file
        else "–"
    )
    _view = [
        mo.hstack(
            [previous_button, image_slider, next_button], align="center", widths=[0, 1, 0]
        ),
        mo.md(f"<span style='font-size:0.85em;opacity:0.7'>{_position}</span>"),
        view_step,
        _pixel_exact(_images_by_step[view_step.value]),
        mo.Html(f'<div style="font-size:0.85em">{_swatches}</div>'),
        mo.md(
            f"<span style='font-size:0.85em;opacity:0.7'>{palette.name} · "
            f"{snapped_share:.0%} av pixlarna helt snäppta</span>"
        ),
        compare_original,
    ]
    if compare_original.value:
        _view.append(_pixel_exact(scaled_image))
    if load_error:
        _view.insert(
            0,
            mo.callout(mo.md(f"**Kunde inte läsa bilden:** {load_error}"), kind="danger"),
        )

    mo.vstack(_view, gap=0.5)
    return


@app.cell(hide_code=True)
def _(PALETTES, palette_dropdown):
    palette = PALETTES[palette_dropdown.value]
    return (palette,)


@app.cell(hide_code=True)
def _(Path, convert, folder_input):
    folder_error = None
    try:
        image_files = convert.find_images(Path(folder_input.value.strip()))
        if not image_files:
            folder_error = "Mappen innehåller inga bilder."
    except OSError as error:
        image_files = []
        folder_error = str(error)
    return folder_error, image_files


@app.cell(hide_code=True)
def _(image_files, mo, set_image_index):
    # Own cell, so that the image slider is redrawn when a button is clicked
    _last_index = max(len(image_files) - 1, 0)

    def _step_button(label, delta):
        def _step(clicks):
            set_image_index(
                lambda index: min(max(min(index, _last_index) + delta, 0), _last_index)
            )
            return (clicks or 0) + 1

        return mo.ui.button(label=label, value=0, on_click=_step)

    previous_button = _step_button("◀", -1)
    next_button = _step_button("▶", 1)
    return next_button, previous_button


@app.cell(hide_code=True)
def _(get_image_index, image_files, mo, set_image_index):
    _last_index = max(len(image_files) - 1, 0)
    image_slider = mo.ui.slider(
        0,
        max(_last_index, 1),
        step=1,
        value=min(get_image_index(), _last_index),
        on_change=set_image_index,
        label="Bild",
        show_value=True,
        debounce=True,
        full_width=True,
    )
    return (image_slider,)


@app.cell(hide_code=True)
def _(
    Image,
    convert,
    folder_error,
    image_files,
    image_slider,
    palette,
    resample_dropdown,
):
    load_error = folder_error
    current_file = None
    _source = Image.new("RGB", palette.size, (128, 128, 128))
    if image_files:
        current_file = image_files[min(image_slider.value, len(image_files) - 1)]
        try:
            _source = Image.open(current_file)
        except OSError as _error:
            load_error = f"{current_file.name}: {_error}"

    scaled_image = convert.fit_to_panel(_source, palette, resample_dropdown.value)
    return current_file, load_error, scaled_image


@app.cell(hide_code=True)
def _(contrast, convert, gamma, saturation, scaled_image):
    toned_image = convert.adjust_tone(
        scaled_image, gamma.value, contrast.value, saturation.value
    )
    return (toned_image,)


@app.cell(hide_code=True)
def _(convert, flatten_threshold, flatten_window, toned_image):
    flattened = convert.flatten(
        toned_image, flatten_threshold.value, flatten_window.value
    )
    return (flattened,)


@app.cell(hide_code=True)
def _(convert, flattened, gamut_strength, palette):
    gamut_mapped = convert.map_gamut(flattened, palette, gamut_strength.value)
    gamut_image = convert.oklab_to_image(gamut_mapped)
    return gamut_image, gamut_mapped


@app.cell(hide_code=True)
def _(convert, gamut_mapped, hue_pull_range, hue_pull_strength, palette, snap_radius):
    pulled, snapped_share = convert.pull_hues_and_snap(
        gamut_mapped,
        palette,
        hue_pull_strength.value,
        hue_pull_range.value,
        snap_radius.value,
    )
    pulled_image = convert.oklab_to_image(pulled)
    return pulled, pulled_image, snapped_share


@app.cell(hide_code=True)
def _(chroma_weight, convert, dither_strength, palette, pulled):
    palette_indices = convert.dither(
        pulled, palette, dither_strength.value, chroma_weight.value
    )
    preview_image = convert.to_preview_image(palette_indices, palette)
    panel_image = convert.to_panel_image(palette_indices, palette)
    return palette_indices, panel_image, preview_image


if __name__ == "__main__":
    app.run()
