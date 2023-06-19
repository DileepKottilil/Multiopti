import tkinter as tk
from tkinter import ttk
import multiopti as mop
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec

#from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
#Dileep

# Function to execute when the "Plot" button is clicked
def plot_result():
    # Create an instance of the multiopti class by Dileep
    mo = mop.multiopti(1, 4, 600) #1 eV to 3 eV only for plotting and extrapolating r.i. values. it has no influence on the ARR
    mo.ref_indx(source='experiment',fileName = "mos3.txt",draw=0)
    #mo.ref_indx(source='theory',draw=0)

    #When there is an interpolation problem, set the multiopti energy values accordingly
    mo.EM(wl = 400,wg = 800, w_step = 1, angle_max = 50, angle_step = 1)

    # Get values from the input fields
    Bragg = float(Bragg_var.get())
    mode = int(mode_var.get())
    air_n = float(air_n_var.get())

    DBR_per_up = float(DBR_per_up_var.get())
    DBR_per_bot = float(DBR_per_bot_var.get())
    
    lr1_n = float(lr1_n_var.get())
    lr2_n = float(lr2_n_var.get())
    cav_n = float(cav_n_var.get())
    lr4_n = float(lr4_n_var.get())
    lr5_n = float(lr5_n_var.get())
    sub_n = float(sub_n_var.get())
    exc_num = int(exc_num_var.get())
    exc_thick = float(exc_thick_var.get())

    # Call the DBR, calc, and plot_reslt methods
    # brg_cav_thick is cavity designed for Braggs wavelength w/o exciton.
    # act_cav_thick is actual cavity length with exciton layers.
    # When no excitons both of these shoudl be same.
    a, b, lr1, lr2, lr4, lr5 = mo.DBR(Bragg, mode, air_n, DBR_per_up, DBR_per_bot,
           lr1_n, lr2_n, cav_n, lr4_n, lr5_n, sub_n,
           exc_num, exc_thick)
    
    
    mo.calc()

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    fig, ax1 = mo.plot_reslt(ax1)
    fig, ax2 = mo.plot_0Deg(ax2, given_inputs = [0,10,50])

    fig, ax3 = mo.DBRplot(ax3)

    

    #fig.tight_layout()  

   

   

     # Clear the right pane
    for widget in right_pane.winfo_children():
        widget.destroy()

    # Display the plot in the right pane
    canvas = FigureCanvasTkAgg(fig, master=right_pane)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0)



    


# Create the main window
root = tk.Tk()
root.title("Multiopti GUI")

# Create the left and right panes
left_pane = ttk.Frame(root)
left_pane.grid(row=0, column=0, sticky="ns")
right_pane = ttk.Frame(root)
right_pane.grid(row=0, column=1)

# Create input fields and labels in the left pane
Bragg_var = tk.StringVar()
mode_var = tk.StringVar()
air_n_var = tk.StringVar()
DBR_per_up_var = tk.StringVar()
DBR_per_bot_var = tk.StringVar()
lr1_n_var = tk.StringVar()
lr2_n_var = tk.StringVar()
cav_n_var = tk.StringVar()
lr4_n_var = tk.StringVar()
lr5_n_var = tk.StringVar()
sub_n_var = tk.StringVar()
exc_num_var = tk.StringVar()
exc_thick_var = tk.StringVar()

inputs = [("Bragg", Bragg_var), ("mode", mode_var), ("air_n", air_n_var),
          ("DBR_per_up", DBR_per_up_var), ("DBR_per_bot", DBR_per_bot_var),
          ("lr1_n", lr1_n_var), ("lr2_n", lr2_n_var), ("cav_n", cav_n_var),
          ("lr4_n", lr4_n_var), ("lr5_n", lr5_n_var), ("sub_n", sub_n_var),
          ("exc_num", exc_num_var), ("exc_thick", exc_thick_var)]

for i, (label_text, var) in enumerate(inputs):
    #label = ttk.Label(left_p
    label = ttk.Label(left_pane, text=label_text)
    label.grid(row=i, column=0, padx=10, pady=5, sticky="w")
    entry = ttk.Entry(left_pane, textvariable=var)
    entry.grid(row=i, column=1, padx=10, pady=5, sticky="w")

# Create the "Plot" button in the left pane
plot_button = ttk.Button(left_pane, text="Plot", command=plot_result)
plot_button.grid(row=len(inputs), columnspan=2, pady=10)

# Set the initial values for the input fields
Bragg_var.set("1550")
mode_var.set("1") #This fixes the Braggs mode as the 1st cavity mode
air_n_var.set("1")
DBR_per_up_var.set("4")
DBR_per_bot_var.set("4")
lr1_n_var.set("1.4")
lr2_n_var.set("2.2")
cav_n_var.set("1")
lr4_n_var.set("2.34")
lr5_n_var.set("1.48")
sub_n_var.set("1.5")
exc_num_var.set("1")
exc_thick_var.set("1")

# Run the main loop
root.mainloop()

