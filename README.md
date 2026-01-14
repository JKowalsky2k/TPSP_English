# TPSP English
Torunski Program Strzelectwa Parkurowego dla Parkur English

## Szybki start (Windows, mało technicznie)

1. **Zainstaluj Python z Microsoft Store.** Wystarczy zwykła instalacja domyślna.
2. **Otwórz PowerShell w folderze projektu.**  
   Najprościej: wejdź do folderu w Eksploratorze plików, kliknij pasek adresu, wpisz `powershell` i Enter.
3. **Zainstaluj potrzebne biblioteki (jednorazowo).** Skopiuj/wklej do PowerShell:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install flask flask_sqlalchemy reportlab
   ```
4. **Uruchom aplikację dwuklikiem.** W folderze projektu kliknij dwa razy plik `start.bat`.
5. **Wejdź w przeglądarce na adres:** `http://127.0.0.1:5000/`

Jeśli okno znika zbyt szybko, otwórz PowerShell (Start → PowerShell), przejdź do folderu projektu poleceniem `cd ŚCIEŻKA_DO_FOLDERU` i uruchom `.\start.bat`, by zobaczyć ewentualne komunikaty.
