# AI-PhotoSearch (Portable)

## 📸 Example Interface

![App Screenshot](screenshot.png)

This application allows you to search for images by text description and/or file name using the CLIP neural network.

No need to install Python or any libraries — a portable Python environment is provided separately.

---

## 🚀 Quick Start

1. **Clone or download this repository** to your computer.
2. **Download the portable Python archive [`system.zip`](https://drive.google.com/open?id=1-_KxftuNN8I320GGNvG2oZjBJodxAERH&usp=drive_fs from Google Drive and unzip it into the project root, so you have a `system/` folder next to `app.py` and `run_app.bat`.
3. Your project folder should now look like this:
    ```
    AI_PhotoSearch/
    ├── app.py
    ├── run_app.bat
    ├── system/
    │   └── python/
    │        ├── python.exe
    │        ├── ... (DLLs, site-packages, etc.)
    ├── cache/                (created automatically)
    ├── last_used_paths.pkl   (created automatically)
    ```
4. **Double-click `run_app.bat`** to launch the application.
    - The app will open in your browser at [http://localhost:8501/](http://localhost:8501/).
    - If it does not open automatically, copy the address from the terminal and paste it into your browser.

---

## 🖼️ How to Use

- **Folder path:** Enter or paste the path to your image folder (Tip: In Windows Explorer, right-click the folder and select “Copy as path”).
- **Image description:** Enter a description for semantic search (using CLIP).
- **File name:** Search by all or part of the file name.
- **Scale:** Adjusts the number of columns for displaying images.
- **Number of results:** Limits the number of images shown.
- **Show all:** Check to show all images in the database.
- **Save:** Save found images to a new folder.

---

## 📄 FAQ

- **No need to install Python or libraries:** Just download and unzip the `system` archive as described above.
- **Folder history** is stored in `last_used_paths.pkl` (auto-generated in the app directory).
- **Image feature cache** is stored in the `cache/` folder (auto-generated).
- **If the app does not launch:** Make sure you run `run_app.bat` and that your system allows running `.bat` files.
- **To clear folder history:** Delete `last_used_paths.pkl` (it will be recreated automatically).
- **Portable:** You can copy the entire folder to another computer and run it the same way (just don’t forget to download and extract the `system` archive).

---

## 🔗 Portable Python Download

> **Download the portable Python environment (`system.zip`) from [Google Drive](https://drive.google.com/open?id=1-_KxftuNN8I320GGNvG2oZjBJodxAERH&usp=drive_fs) and unzip it into the project root.**
>
> You do **not** need to install Python or any dependencies — everything required is included in the `system` folder.

---

## ✉️ Contact

If you have questions, contact the script author: romanlosev@gmail.com

---

**Enjoy using AI-PhotoSearch!**
