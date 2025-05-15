# Tesseract OCR Installation Guide for Windows

This guide will help you install Tesseract OCR on your Windows machine, which is required for the document classification service to extract text from images and PDFs.

## Installation Steps

1. **Download the Tesseract installer**:
   - Visit the [UB Mannheim Tesseract GitHub page](https://github.com/UB-Mannheim/tesseract/releases)
   - Download the latest version (e.g., `tesseract-ocr-w64-setup-5.3.0.20221222.exe`)
   - Or use this direct link: [Tesseract 5.3.0 (64-bit)](https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.0.20221222/tesseract-ocr-w64-setup-5.3.0.20221222.exe)

2. **Run the installer**:
   - Double-click the downloaded file to start the installation
   - Accept the license agreement
     - French
     - Arabic 
   - This ensures support for all required languages in your document classification service
   - Click "Install"

3. **Add Tesseract to PATH** (if not done automatically during installation):
   - Right-click on "This PC" or "My Computer" and select "Properties"
   - Click on "Advanced system settings"
   - Click on "Environment Variables"
   - Under "System variables", find and select the "Path" variable, then click "Edit"
   - Click "New" and add the path to the Tesseract installation directory (typically `C:\Program Files\Tesseract-OCR`)
   - Click "OK" on all dialogs to save changes

4. **Verify installation**:
   - Open Command Prompt (cmd)
   - Type `tesseract --version` and press Enter
   - You should see version information if Tesseract is installed correctly

5. **Update config.py**:
   - If Tesseract is in your PATH, you can use:
     ```python
     TESSERACT_PATH = "tesseract"
     ```
   - If you need to specify the full path:
     ```python
     TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
     ```

6. **Install necessary language data packages**:
   - During the Tesseract installation, make sure you select the additional language data (French, and Arabic)
   - If you didn't, run the installer again and modify the installation to add these languages

## Troubleshooting

If you see errors like:

```
Error extracting text from image: C:\Program Files\Tesseract-OCR\tesseract.exe is not installed or it's not in your PATH.
```

This means the application can't find the Tesseract executable. Check the following:

1. Confirm Tesseract is installed correctly
2. Verify the path in `config.py` matches your installation location
3. Try using the absolute path to the executable
4. Restart your application after making changes to ensure they take effect

## Next Steps

After installing Tesseract, restart your application. The document classification service should now be able to extract text properly from images and PDFs. 