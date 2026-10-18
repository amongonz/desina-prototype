"""Small local desktop editor. No server, GPU, pretrained weights, or mesh required."""
from pathlib import Path
import copy
import numpy as np
from .core import Camera, Model, brush, load_project, reconstruct, render


def launch(path):
    import tkinter as tk
    from tkinter import ttk, colorchooser, filedialog, messagebox
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title('Desina / implicit sketch studio')
    root.configure(bg='#18262d')
    model = Model.load(path)
    history = []
    paint = [.95, .24, .35]
    hit_points = None
    hit_mask = None
    preview_size = 160
    display_size = 560
    pending = None
    azimuth = tk.DoubleVar(value=30)
    elevation = tk.DoubleVar(value=20)
    radius = tk.DoubleVar(value=.14)
    mode = tk.StringVar(value='paint')
    status = tk.StringVar(value=f'{Path(path).name} / click surface to apply a 3D brush')
    frame = ttk.Frame(root, padding=18)
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='DESINA  /  Drawing into volume', font=('', 18, 'bold')).grid(
        row=0, column=0, columnspan=2, sticky='w', pady=(0, 14))
    canvas = tk.Canvas(frame, width=display_size, height=display_size, highlightthickness=0)
    canvas.grid(row=1, column=0, rowspan=10)
    controls = ttk.Frame(frame, padding=(20, 0))
    controls.grid(row=1, column=1, sticky='new')

    def redraw():
        nonlocal hit_points, hit_mask, pending
        if pending:
            root.after_cancel(pending)
        pending = None
        rgb, hit_mask, hit_points = render(model, Camera(azimuth.get(), elevation.get()), preview_size)
        img = Image.fromarray(np.uint8(rgb*255)).resize((display_size, display_size), Image.Resampling.BILINEAR)
        canvas.image = ImageTk.PhotoImage(img)
        canvas.delete('all')
        canvas.create_image(0, 0, anchor='nw', image=canvas.image)

    def schedule(_=None):
        nonlocal pending
        if pending:
            root.after_cancel(pending)
        pending = root.after(250, redraw)

    def slider(label, variable, low, high, command=None):
        ttk.Label(controls, text=label).pack(anchor='w', pady=(12, 3))
        ttk.Scale(controls, from_=low, to=high, variable=variable, command=command).pack(fill='x')

    slider('Orbit / azimuth', azimuth, -180, 180, schedule)
    slider('Elevation', elevation, -80, 80, schedule)
    ttk.Label(controls, text='Click brush', font=('', 12, 'bold')).pack(anchor='w', pady=(25, 8))
    for name in ['paint', 'add', 'subtract', 'smooth']:
        ttk.Radiobutton(controls, text=name.title(), variable=mode, value=name).pack(anchor='w')
    slider('Brush radius / world units', radius, .04, .35)

    def choose_color():
        nonlocal paint
        rgb, _ = colorchooser.askcolor(color=tuple(int(c*255) for c in paint), parent=root)
        if rgb:
            paint = [c/255 for c in rgb]

    ttk.Button(controls, text='Choose paint color', command=choose_color).pack(fill='x', pady=12)

    def stroke(event):
        if pending:  # Never pick against a camera whose preview has not finished rendering.
            redraw()
        x, y = int(event.x*preview_size/display_size), int(event.y*preview_size/display_size)
        if not (0 <= x < preview_size and 0 <= y < preview_size and hit_mask[y, x]):
            status.set('Click on the visible surface; rotate to reach another side.')
            return
        history.append((model.sdf.copy(), model.color.copy(), copy.deepcopy(model.metadata)))
        if len(history) > 8:
            history.pop(0)
        brush(model, hit_points[y, x], radius.get(), mode.get(), paint)
        status.set(f'{mode.get().title()} stroke / {len(model.metadata.get("edits", []))} edits / save to keep changes')
        redraw()

    def undo():
        if history:
            model.sdf, model.color, model.metadata = history.pop()
            status.set('Undid last brush stroke')
            redraw()

    def save():
        name = filedialog.asksaveasfilename(parent=root, defaultextension='.npz',
                                           initialfile='edited.npz', filetypes=[('Desina volume', '*.npz')])
        if name:
            try:
                model.save(name)
                status.set(f'Saved {name}')
            except OSError as error:
                messagebox.showerror('Save failed', str(error))

    def open_project():
        nonlocal model, path
        name = filedialog.askopenfilename(parent=root, filetypes=[('Drawing project', '*.json')])
        if name:
            status.set('Reconstructing drawings and baking surface paint…')
            root.update_idletasks()
            try:
                views, _ = load_project(name)
                result = reconstruct(views)
                model = result
                path = name
                history.clear()
                status.set('Reconstructed project / save volume to keep it')
                redraw()
            except (ValueError, OSError, KeyError) as error:
                messagebox.showerror('Reconstruction failed', str(error))
                status.set('Could not reconstruct project')

    ttk.Button(controls, text='Undo brush', command=undo).pack(fill='x', pady=4)
    ttk.Button(controls, text='Save volume as…', command=save).pack(fill='x', pady=4)
    ttk.Button(controls, text='Reconstruct drawings…', command=open_project).pack(fill='x', pady=4)
    ttk.Label(controls, text='One click = one 3D stroke.\nRotate to inspect the result.\nUndo keeps the last 8 strokes.\n\nNew drawing reconstructions\nstart a fresh volume.', justify='left').pack(anchor='w', pady=18)
    ttk.Label(frame, textvariable=status, wraplength=780).grid(row=11, column=0, columnspan=2, sticky='w', pady=(12, 0))
    canvas.bind('<Button-1>', stroke)
    root.bind('<Control-z>', lambda _: undo())
    redraw()
    root.mainloop()
