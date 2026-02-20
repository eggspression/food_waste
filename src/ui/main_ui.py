import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import os
import cv2
from PIL import Image, ImageTk



from src.pipeline.plate_prep import plate_seg_no_depth
from src.pipeline.food_seg_ai import food_seg
from src.pipeline.vol_seg_ai import volume_cal
from src.pipeline.vol_tot import show_total_volume



class ResultWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Résultat de segmentation")
        self.geometry("2560x1440")
        self.configure(bg="lightgray")

        self.rgb_path = master.rgb_image_path
        self.depth_path = master.depth_image_path
        
        self.mode = master.operation_mode.get()
        # self.method = master.seg_method.get()
        self.plate_only = None


        self.create_widgets()

    def create_widgets(self):
        title = tk.Label(self, text="Page de résultats", font=("Segoe UI", 18), bg="lightgray")
        title.pack(pady=10)

        
        self.image_label = tk.Label(self, bg="white")
        self.image_label.pack(pady=10)

        
        button_frame = tk.Frame(self, bg="lightgray")
        button_frame.pack(pady=10)
        if self.mode == "seg" or self.mode == "both": 
            self.btn_plate = tk.Button(button_frame, text="Segmentation des assiettes",font= ("Segoe UI",10), command=self.plate_seg_ai)
            self.btn_plate.grid(row=0, column=0, padx=10)

            self.btn_food = tk.Button(button_frame, text="Segmentation des aliments",font= ("Segoe UI",10), command=self.food_seg_ai, state="disabled")
            self.btn_food.grid(row=0, column=1, padx=10)
            if self.mode == "both":
                self.btn_ir = tk.Button(button_frame, text = "Graphique de profondeur", font =("Segoe UI",10), command= self.vol_show, state = "disabled" )
                self.btn_vol = tk.Button(button_frame, text="Calcul du volume",font= ("Segoe UI",10), command=self.volume_calc_ai, state="disabled")
                self.btn_ir.grid(row = 0, column = 2, padx = 10)
                self.btn_vol.grid(row=0, column=3, padx=10)
        if self.mode == "vol":
            self.btn_vol = tk.Button(button_frame, text="Calcul du volume",font= ("Segoe UI",10), command=self.volume_calc_ai)
            self.btn_vol.grid(row=0, column=2, padx=10)
     
        if self.mode == "vol" or self.mode == "both":
            self.volume_text = tk.Label(self, text="Volume Résultat : Pas encore calculé", font=("Segoe UI", 15), bg="lightgray")
            self.volume_text.pack()

       
        self.current_image = None
        self.update_preview(self.rgb_path)

    def update_preview(self, path_or_array):
        try:
            if isinstance(path_or_array, str):
                img = Image.open(path_or_array)
            else:
                img = Image.fromarray(path_or_array)
            img = img.resize((1280,720))
            self.current_image = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.current_image)
        except Exception as e:
            messagebox.showerror("Error", e)

    

    def plate_seg_ai(self):
        try:
            img = cv2.imread(self.rgb_path)
            self.plate_only, plate_only_mask = plate_seg_no_depth(self.rgb_path)
            plate_display = cv2.cvtColor(self.plate_only, cv2.COLOR_BGR2RGB)
            plate_display_resized = cv2.resize(plate_display, (img.shape[1], img.shape[0]))
            self.update_preview(plate_display_resized)
            self.btn_food.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Error", e)
    
    def food_seg_ai(self):
        try:
            img = cv2.imread(self.rgb_path)
            fig = food_seg(self.plate_only)
            fig = cv2.cvtColor(fig, cv2.COLOR_BGR2RGB)
            fig_resized = cv2.resize(fig, (img.shape[1], img.shape[0]))
            self.update_preview(fig_resized)
            if self.mode == "both" or self.mode == "vol":
                self.btn_vol.configure(state="normal")
                self.btn_ir.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Error", e)

    def volume_calc_ai(self):
        volume = volume_cal(self.rgb_path, self.depth_path)
        labels = ["Feculent", "Viande", "Legumes", "Dechets"]
        self.volume_text.configure(text=f"Le volume pour chaque type est {', '.join([f'{labels[i]}: {volume[i]:.2f} mm³' for i in range(len(volume))])}")

    def vol_show(self):
        try:
            img = cv2.imread(self.rgb_path)
            fig = show_total_volume(self.rgb_path, self.depth_path)
            # fig = cv2.cvtColor(fig, cv2.COLOR_BGR2RGB)
            # fig_resized = cv2.resize(fig, (img.shape[1], img.shape[0]))
            self.update_preview(fig)

        except Exception as e:
            messagebox.showerror("Error", e)


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Outil d'Estimation du Gaspillage Alimentaire")
        self.geometry("2560x1440")
        self.configure(bg="white")
        

        self.input_method = tk.StringVar(value="load")
        # self.seg_method = tk.StringVar(value="classic")
        self.operation_mode = tk.StringVar(value = "seg")

        self.rgb_image_path = None
        self.depth_image_path = None
      
        self.create_widgets()




    def create_widgets(self):
        title = tk.Label(self, text= "Bienvenue dans l'outil d'estimation du gaspillage alimentaire", font = ("Segoe UI",40),bg="white")
        title.pack(pady=70)

        description = tk.Label(self, text = "Cet outil permet de calculer le gaspillage alimentaire restant. Vous devez prendre ou télécharger une ou deux images (selon le mode choisi).\n"
                                            "L'une est l'image RVB et l'autre est l'image de profondeur (requise uniquement pour le mode volume ou segmentation + volume).\n" \
                                            " Vous pouvez choisir entre deux méthodes de segmentation :\n "
                                            "- La méthode classique : segmentation avec des outils appris des TD (skimage, cv2, etc..)\n" \
                                            "- La méthode IA : segmentation à l'aide du modèle de Machine Learning que nous avons développé"
                                            
                                            ,font = ("Segoe UI",14), bg="white")
                            
        description.pack()
       

        text1 = tk.Label(self, text = 'Veuillez choisir le mode de fonctionnement souhaité, la saisie et la méthode de segmentation', font = ("Segoe UI",20),bg="white")
        text1.pack(pady=20)
        #Frame for operation mode
        mode_frame = tk.Frame(self, bg="white")
        mode_frame.pack(pady=10)

        mode_label = tk.Label(mode_frame, text="Mode de fonctionnement :", font=("Segoe UI", 20), bg="white")
        mode_label.grid(row=0, column=0, padx=10)
        
        seg_radio = ttk.Radiobutton(mode_frame, text="Segmentation uniquement", variable=self.operation_mode, value="seg",style="Custom.TRadiobutton")
        vol_radio = ttk.Radiobutton(mode_frame, text="Volume uniquement", variable=self.operation_mode, value="vol",style="Custom.TRadiobutton")
        vol_and_seg_radio = ttk.Radiobutton(mode_frame, text="Segmentation + Volume", variable=self.operation_mode, value="both", style="Custom.TRadiobutton")
        
        seg_radio.grid(row=0, column=1, padx=20)
        # vol_radio.grid(row=0, column=2, padx=20)
        vol_and_seg_radio.grid(row=0, column=3, padx=20)


        #Frame for input method
        input_frame = tk.Frame(self, bg="white")
        input_frame.pack(pady=10)

        input_label = tk.Label(input_frame, text="Input:", font=("Segoe UI", 20), bg="white")
        input_label.grid(row=0, column=0, padx=10)
        
        load_radio = ttk.Radiobutton(input_frame, text="Telecharger l'image", variable=self.input_method, value="load",style="Custom.TRadiobutton")
        take_radio = ttk.Radiobutton(input_frame, text="Prendre l'image", variable=self.input_method, value="take", style="Custom.TRadiobutton")
        load_radio.grid(row=0, column=1, padx=20)
     

        load_btn = ttk.Button(self, text="Telecharger",style="Primary.TButton", command=self.handle_load_btn)
        load_btn.pack(pady=5)


        next_btn = ttk.Button(self, text="Suivante", style="Primary.TButton", command=self.confirm_choices)
        next_btn.pack(pady=15)

    def handle_load_btn(self):
        if self.input_method.get() == "load":
            rgb_path = filedialog.askopenfilename(title="Sélectionnez l'image RVB",
                                                  filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
            
            if not rgb_path:
                messagebox.showwarning("Aucun fichier", "Aucune image RVB sélectionnée.")
                return
            
            self.rgb_image_path = rgb_path
            if self.operation_mode.get() == "vol" or self.operation_mode.get() == "both":
                depth_path = filedialog.askopenfilename(title="Sélectionner la matrice de profondeur",
                                                        filetypes=[("Depth Files", "*.png *.npy")])
                if not depth_path:
                    messagebox.showwarning("Aucun fichier", "Aucune matrice de profondeur sélectionnée.")
                    return
                self.depth_image_path = depth_path

            messagebox.showinfo("Fichiers chargés", "Vos fichiers ont été téléchargés avec succès")
        
        else:
            messagebox.showinfo("Camera", "La capture de caméra n'est pas encore implémentée.")
    
    def confirm_choices(self):
        try:
            img = Image.open(self.rgb_image_path)
            img = img.resize((640, 360))
            tk_img = ImageTk.PhotoImage(img)

            confirm_window = tk.Toplevel(self)
            confirm_window.title("Confirmer l'image")
            confirm_window.geometry("700x500")
            confirm_window.configure(bg="white")

            tk.Label(confirm_window, text="Est-ce la bonne image ?", font=("Segoe UI", 14), bg="white").pack(pady=10)
            img_label = tk.Label(confirm_window, image=tk_img)
            img_label.image = tk_img  
            img_label.pack(pady=10)

            def confirm():
                confirm_window.destroy()
                ResultWindow(self)

            def cancel():
                self.rgb_image_path = None
                self.depth_image_path = None
                confirm_window.destroy()
                messagebox.showinfo("Reset", "Les chemins d'entrée ont été effacés. Veuillez resélectionner vos fichiers.")

            btn_frame = tk.Frame(confirm_window, bg="white")
            btn_frame.pack(pady=20)

            tk.Button(btn_frame, text="Oui", width=10, command=confirm).grid(row=0, column=0, padx=10)
            tk.Button(btn_frame, text="Non", width=10, command=cancel).grid(row=0, column=1, padx=10)

        except Exception as e:
            messagebox.showerror("Error", e)

def main():
    app = MainWindow()
    app.tk.call('source', 'theme/breeze.tcl') 
    style = ttk.Style(app)

    style.theme_use('Breeze')
    style.configure("Custom.TRadiobutton", font=("Segoe UI", 15), background="white")
    style.configure(
        "Primary.TButton",        
        font=("Segoe UI", 15),
        foreground="white",
        background="#007acc",
        padding=10,
        relief="flat"
    )
    style.map(
        "Primary.TButton",
        background=[
            ("active",  "#3399ff"),
            ("pressed", "#005a9e")
        ],
        foreground=[("disabled", "#d9d9d9")]
    )
    app.mainloop()


if __name__ == "__main__":
    main()