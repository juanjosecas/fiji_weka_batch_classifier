#@ String (visibility=MESSAGE, value="<html><h2 style='color:blue'>Weka Classification Batch Processor</h2><p style='font-size:14px;'>Este procesador clasifica imágenes usando un clasificador entrenado de Weka y cuenta blobs según el área mínima especificada para cada clase definida por el clasificador. Los resultados se guardan en un archivo CSV y las imágenes segmentadas se pueden guardar en una carpeta separada.</p><br/><ul style='font-size:12px;'><li>Clasifica imágenes de entrada usando un clasificador Weka</li><li>Cuenta blobs de tamaño mínimo para cada clase</li><li>Guarda los resultados en una tabla CSV</li><li>Guarda las imágenes segmentadas y la visualización de detección de blobs en carpetas separadas</li></ul><br/>* = Obligatorio</html>") docmsg
#@ File[] (label="Imágenes de entrada*", required=true, style="open") input_files  # Imágenes de entrada
#@ String (visibility=MESSAGE, value="", required=false) only_for_visual
#@ File(label="Clasificador*", required=true, style="open") classifier_file  # Archivo del clasificador Weka
#@ Integer(label="Área mínima del blob", required=true, value=0, stepSize=1) min_particle_area  # Área mínima de partículas
#@ String (visibility=MESSAGE, value="", required=false) only_for_visual2
#@ Boolean(label="Mostrar imágenes segmentadas", required=true, value=False) show_segmented  # Mostrar imágenes segmentadas
#@ Boolean(label="Mostrar/guardar imágenes de detección de blobs", value=True) show_or_save_blob_images  # Mostrar/guardar imágenes de detección de blobs
#@ File(label="Guardar imágenes segmentadas (carpeta)", style="directory", required=false) segmented_folder  # Carpeta para guardar imágenes segmentadas
#@ String (visibility=MESSAGE, value="", required=false) only_for_visual3
#@ File(label="Guardar archivo de resultados (CSV)", style="save", required=false) result_file  # Archivo CSV para guardar resultados
#@ Boolean(label="Adjuntar configuración al archivo de resultados", description="Guardar la configuración utilizada para la clasificación para referencia futura", value=true) append_settings  # Adjuntar configuración a los resultados

import os
from trainableSegmentation import WekaSegmentation
from trainableSegmentation.utils.Utils import getGoldenAngleLUT
from ij import IJ
from ij.plugin import Duplicator, ImageCalculator
from ij.plugin.filter import ParticleAnalyzer
from ij.process import ImageStatistics
from ij.measure import ResultsTable, Measurements

def main():
	global input_file, classifier_file, min_particle_area, show_segmented, result_file, append_settings, segmented_folder

	# Verificar si el archivo de resultados ya existe
	if result_file is not None:
		if os.path.exists(result_file.getAbsolutePath()):
			if not IJ.showMessageWithCancel("Advertencia", "El archivo de resultados ya existe. ¿Desea continuar?"):
				return

	# Verificar si la carpeta de imágenes segmentadas está vacía
	if segmented_folder is not None:
		if len(os.listdir(segmented_folder.getAbsolutePath())) != 0:
			if not IJ.showMessageWithCancel("Advertencia", "La carpeta de salida de imágenes segmentadas no está vacía. ¿Desea continuar?"):
				return
	
	# Cargar el clasificador Weka
	weka_seg = WekaSegmentation()
	if not weka_seg.loadClassifier(classifier_file.getAbsolutePath()):
		IJ.error("No se pudo cargar el clasificador")
		return
	if weka_seg.getTrainHeader().numAttributes() < 1:
		IJ.error("El encabezado de entrenamiento del clasificador no tiene atributos. No estoy seguro de lo que esto significa, tal vez no se ejecutó el entrenamiento.")
		return
		
	num_classes = weka_seg.getNumOfClasses()
	class_labels = weka_seg.getClassLabels()
	
	# Crear tabla para almacenar los resultados de la clasificación
	classification_table = ResultsTable()
	classification_table.show("Resultados de Clasificación")
	
	# Procesar cada imagen de entrada
	for input_index, input_file in enumerate(input_files):
		input_file_path = input_file.getAbsolutePath()
		input_image = IJ.openImage(input_file_path)
		image_title = input_image.getTitle()
		
		# Aplicar clasificador a la imagen
		IJ.log("Clasificando imagen {}".format(image_title))
		classified = weka_seg.applyClassifier(input_image)

		# Aplicar tabla de colores para hacer visibles las clases
		classified_vis = Duplicator().run(classified)
		lut = getGoldenAngleLUT()
		classified_vis.setLut(lut)
		IJ.run(classified_vis, "RGB Color", "")
		IJ.run(classified_vis, "Apply LUT", "")
		classified_vis.setTitle(image_title + " Clasificación")

		# Mostrar imágenes segmentadas si la opción está habilitada
		if show_segmented:
			input_image.show()
			classified_vis.show()

		# Guardar imágenes segmentadas si se especificó una carpeta de salida
		if segmented_folder is not None:
			segmented_path = os.path.join(segmented_folder.getAbsolutePath(), os.path.splitext(image_title)[0] + "-segmentado.tif")
			IJ.saveAs(classified_vis, "TIFF", segmented_path)
		
		# Utilizar histograma para calcular el área por clase
		histogram = ImageStatistics.getStatistics(classified.getProcessor()).getHistogram()
		image_pixels = input_image.getWidth() * input_image.getHeight()
		
		# Crear ParticleAnalyzer para contar y analizar blobs
		ParticleAnalyzer.setSummaryTable(classification_table)  # Método estático para todos los analizadores de partículas
		particle_analyzer = ParticleAnalyzer(
			ParticleAnalyzer.SHOW_OUTLINES | ParticleAnalyzer.DISPLAY_SUMMARY,
			Measurements.AREA,
			None,
			min_particle_area, 1e100)
		particle_analyzer.setHideOutputImage(True)
		
		# Analizar cada clase
		for i in range(num_classes):
			IJ.log("Ejecutando analizador de partículas en clase {}".format(class_labels[i]))
		
			# Seleccionar una clase ajustando el umbral mínimo y máximo a este valor de clase
			IJ.setRawThreshold(classified, i, i, "Red")
		
			# Contar blobs utilizando ParticleAnalyzer
			particle_analyzer.analyze(classified)

			# Mostrar/guardar imágenes de detección de blobs si la opción está habilitada
			if show_or_save_blob_images:
				# Imagen de visualización del analizador de partículas
				particles_image = particle_analyzer.getOutputImage()
	
				# Superponer imagen de partículas sobre la original
				IJ.run(particles_image, "Invert LUT", "")
				IJ.run(particles_image, "Apply LUT", "")
				ic = ImageCalculator()
				overlay_image = ic.run("Transparent-zero create", input_image, particles_image)
				overlay_image.setTitle(image_title + " Blobs de la clase: " + class_labels[i])
	
				# Mostrar blobs detectados
				if show_segmented:
					overlay_image.show()
	
				# Guardar imágenes de blobs detectados
				if segmented_folder is not None:
					blobs_path = os.path.join(segmented_folder.getAbsolutePath(), os.path.splitext(image_title)[0] + "-blobs-" + class_labels[i] + ".tif")
					IJ.saveAs(overlay_image, "TIFF", blobs_path)
				
			# Agregar resultados a la tabla de clasificación
			classification_table.addValue("Clase", class_labels[i])
		
			# El área calculada por ParticleAnalyzer excluye blobs demasiado pequeños
			# Adicionalmente, calcular el área total con blobs pequeños
			classification_table.addValue("Archivo", image_title)
			classification_table.addValue("Área Completa", histogram[i])
			classification_table.addValue("%Área Completa", float(histogram[i]) / image_pixels * 100)
		
		# Actualizar la tabla de resultados
		classification_table.show("Resultados de Clasificación")
	
	# Renombrar métricas de partículas para evitar confusión con el área completa
	classification_table.renameColumn("Slice", "Clase")
	classification_table.renameColumn("Total Area", "Área Total de Partículas")
	classification_table.renameColumn("Average Size", "Tamaño Promedio de Partículas")
	classification_table.renameColumn("%Area", "%Área de Partículas")
	classification_table.renameColumn("Count", "Conteo de Partículas")
	
	# Actualizar la tabla de resultados nuevamente
	classification_table.show("Resultados de Clasificación")
	
	# Guardar archivo de resultados si se especificó
	if result_file is not None:
		classification_table.saveAs(result_file.getAbsolutePath())
		# Adjuntar configuración utilizada si la opción está habilitada
		if append_settings:
			with open(result_file.getAbsolutePath(), "a") as result_file:
				result_file.write("\n")
				settings = {
					"Configuración": "Valor",
					"Archivo del clasificador": classifier_file,
					"Área mínima del blob": min_particle_area
				}
				for k, v in settings.items():
					result_file.write("\"{}\",\"{}\"\n".format(k, v))
	
	IJ.showMessage("Clasificación por lotes completada")

main()
