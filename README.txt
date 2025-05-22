# AI PhotoSearch by Description (Portable)

This application allows you to search for images by text description and/or file name using the CLIP neural network.  
Runs fully offline — you don't need to install Python or any libraries (portable Python is included).

---

## How to launch

1. Download and unzip the entire folder (e.g., AI_PhotoSearch) anywhere on your computer.
2. The folder must contain:
   - app.py — main application script
   - run_app.bat — file to run the app (double-click)
   - system/ — portable Python with all required libraries
   - (optional) cache/ — CLIP feature cache (created automatically)
   - (optional) last_used_paths.pkl — recent folder history (created automatically)

3. Double-click run_app.bat to launch.
   - The app will open in your browser.
   - If it does not open, copy the link from the console (usually http://localhost:8501/) and paste it in your browser manually.

---

## How to use

- **Folder path** — paste or type the path to your image folder (in Windows Explorer, right-click the folder and select “Copy as path”).
- **Image description** — enter a description for semantic search (CLIP will be used).
- **File name** — you can search by (a part of) the file name.
- **Scale** — number of columns to display.
- **Number of results** — how many results to show.
- **Show all** — show all images in the database.
- **Save** — save found images to a new folder.

---

## FAQ

- **No need to install Python or libraries** — everything is bundled in the system/ folder.
- **Folder history** is stored in last_used_paths.pkl (in the app directory).
- **Image database cache** is stored in the cache/ folder.
- **If the app does not launch:** make sure you use run_app.bat and your system allows running .bat files.
- **Clear folder history:** delete last_used_paths.pkl (it will be recreated automatically).
- **Portable:** you can copy the entire folder to another computer and run it the same way.

---

## Example folder structure

AI_PhotoSearch_Stable_With_Manual/
├── app.py  
├── run_app.bat  
├── /system/  
│   └── /python/  
│        ├── python.exe  
│        ├── ... (DLLs, site-packages, etc.)  
├── /cache/                (created automatically)  
├── last_used_paths.pkl    (created automatically)  

---

**If you have questions, contact the script author romanlosev@gmail.com. Enjoy!**
