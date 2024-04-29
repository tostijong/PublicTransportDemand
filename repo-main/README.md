# Public Transport Demand and Network Planning and Operations (CIEQ6232) - TU Delft

For any questions regarding this software contact Arjan de Ruijter (A.J.F.deRuijter@tudelft.nl)

----

## Installation
We provide two alternative options to run this software: you can locally install the software in your personal machine or you can use a Virtual Machine (VM) with Ubuntu and the dependencies already set up.

**Please note that if you are using the students' computer labs @ TU Delft you must use the VM, since you don't have admin privileges to install the dependencies locally on those machines**

### Local installation
If you want to run the code locally in your computer, you need to follow these instructions.

**Support for macOS is not provided.**

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html). 

2. Install dependencies:
    - **Only for Windows users**. Download the latest Microsoft Visual C++ Build Tools: 
       * You can download it from [here](https://visualstudio.microsoft.com/downloads/). 
       * In the installation options select "Desktop Development with C++". 
    - **Only for Linux users**. You need GCC compiler (in Ubuntu, install it with `sudo apt install build-essential`)
3. Create the conda environment that installs all the necessary dependencies

   - Open Anaconda Prompt (in Windows) or a terminal (Linux).

   - Move into the directory where this README file is located.
   
       * Windows: use `cd` to move to the target directory. For example, `cd C:\Users\user\Documents\repo-main`. If you have many disk partitions, you need to first move to the partition where the file is located, for example, 'D:' and then `cd D:\Documents\repo-main`.
   	
       * Linux: use the command `cd` in the terminal (e.g., `cd /home/user/repo-main`)
        
   - Run the following command: `conda env create -f environment.yml` 
   
4. Run the Jupyter notebook
   - Activate the newly created environment with the following command: `conda activate pt-networks`
   - Start a new jupyter instance with the following command: `jupyter notebook`
   - A new window will open in your web browser. Select the file notebook.ipynb from the list and start working on your assignment :)

### Using a VM 

#### In the students' computer labs @ TU Delft

1. Run **Haskell VM Import CIEQ6232** from the start menu to load the VM image. Press any key when the script finishes to exit.
2. Run **Oracle VM VirtualBox** from the start menu, select the machine cieq6232 and go to the Settings->USB and select the option USB 1.1 (if not already selected)
3. Run the VM with the green start button.
4. The login details for Ubuntu in the VM are:
    - user: cieq6232
    - pass: cieq6232
5. Within the VM, open a terminal from the left side panel. In the terminal, move to the directory with the code `cd ~/Documents/repo-main`, and follow step 4 in the local installation instructions to run the code. 

Note: you may want to adjust the screen resolution to fit the screen better. For this, you can right click in the desktop and select Display Settings.   

#### Running the VM in your own device

1. Install [Virtual Box](https://www.virtualbox.org/)
2. Download the VM from [here](https://surfdrive.surf.nl/files/index.php/s/Ilj0zuk4kiWMeNj)
3. Import the VM to Virtual Box and Start it (more info [here](https://docs.oracle.com/cd/E26217_01/E26796/html/qs-import-vm.html))
4. The login details for Ubuntu in the VM are:
    - user: cieq6232
    - pass: cieq6232
5. Within the VM, open a terminal from the left side panel. In the terminal, move to the directory with the code `cd ~/Documents/repo-main`, and follow step 4 in the local installation instructions to run the code.    

Note: you may want to adjust the screen resolution to fit the screen better. For this, you can right click in the desktop and select Display Settings.   

----

## Using the code

You can find a video tutorial on how to use this tool [here](https://surfdrive.surf.nl/files/index.php/s/Td4xD7GIDDefniP)
