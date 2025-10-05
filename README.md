Dieses Projekt zeigt die Entwicklung einer einfachen 3D-Rekonstruktions-Pipeline für medizinische CT.
Ziel ist es, aus einer Serie von 2D-Schnittbildern ein 3D-Volumen zu erzeugen als 3D-Modell zu visualisieren.

Projektziele
Laden und Sortieren von DICOM-Dateien
Aufbau eines 3D-Volumens aus 2D-Schnitten
Segmentierung

Verwendete Technologien

Python 3.x
pydicom :zum Laden medizinischer DICOM-Bilder
NumPy :zur numerischen Verarbeitung und Volume-Erzeugung
PyVista – zur 3D-Visualisierung und Mesh-Generierung

Projektstruktur
med3d-project/
│
├── src/
│   ├── load_dicom.py         
│   ├── segment_lung.py      
│   ├── reconstruct_3d.py     
│   └── pipeline.py            
│
├── results/
│   ├── mesh_snapshot.png      
│   └── segmentation_overlay.png
│


Ausführung
# Virtuelle Umgebung aktivieren 
venv\Scripts\activate

# Pipeline starten
python src/pipeline.py


Das Ergebnis wird als interaktive 3D-Darstellung angezeigt und im results/-Ordner gespeichert.
