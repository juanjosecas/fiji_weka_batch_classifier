#@ String (visibility=MESSAGE, value="<html>Weka Classification Batch Processor<br/><ul><li>Runs a classifier on a list of images</li><li>Counts blobs of minimum size for each class defined by the classifier</li><li>Saves the per class count and area in a csv table</li><li>Saves segmentation result and blob detection visualization in a separate folder</li></ul><br/>* = Required</html>") docmsg
#@ File[] (label="Input images*", required=true, style="open") input_files
#@ String (visibility=MESSAGE, value="", required=false) only_for_visual
#@ File(label="Classifier*", required=true, style="open") classifier_file
#@ Integer(label="Minimum blob area", required=true, value=0, stepSize=1) min_particle_area
#@ String (visibility=MESSAGE, value="", required=false) only_for_visual2
#@ Boolean(label="Show segmented images", required=true, value=False) show_segmented
#@ Boolean(label="Also show / save blob detection images", value=True) save_blob_images
#@ File(label="Save segmented images (folder)", style="directory", required=false) segmented_folder
#@ String (visibility=MESSAGE, value="", required=false) only_for_visual3
#@ File(label="Save result file (CSV)", style="save", required=false) result_file
#@ Boolean(label="Append settings to result file", description="Save the settings used for classification for future reference", value=true) append_settings
#@ UIService ui

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

	if result_file is not None:
		if os.path.exists(result_file.getAbsolutePath()):
			if not IJ.showMessageWithCancel("Warning", "Result file exists an. Continue?"):
				return

	if segmented_folder is not None:
		if len(os.listdir(segmented_folder.getAbsolutePath())) != 0:
			if not IJ.showMessageWithCancel("Warning", "Segmented images output folder is not empty. Continue?"):
				return
	
	# Load Weka Classifier
	weka_seg = WekaSegmentation()
	if not weka_seg.loadClassifier(classifier_file.getAbsolutePath()):
		IJ.error("Could not load classifier")
		return
	if weka_seg.getTrainHeader().numAttributes() < 1:
		IJ.error("Classifier train header has no attributes. I'm not sure what this means, maybe training was not run?")
		return
		
	num_classes = weka_seg.getNumOfClasses()
	class_labels = weka_seg.getClassLabels()
	
	# Create table for results
	classification_table = ResultsTable()
	classification_table.show("Classification Results")
	
	for input_index, input_file in enumerate(input_files):
		input_file_path = input_file.getAbsolutePath()
		imp = IJ.openImage(input_file_path)
		image_title = imp.getTitle()
		
		# Apply classifier to image
		IJ.log("Classifying image {}".format(imp.getTitle()))
		classified = weka_seg.applyClassifier(imp)

		# Apply lookup table to image to make classes visible
		classified_vis = Duplicator().run(classified)
		lut = getGoldenAngleLUT()
		classified_vis.setLut(lut)
		IJ.run(classified_vis, "RGB Color", "");
		IJ.run(classified_vis, "Apply LUT", "");

		if show_segmented:
			ui.show(classified_vis)

		if segmented_folder is not None:
			segmented_path = os.path.join(segmented_folder.getAbsolutePath(), os.path.splitext(image_title)[0] + "-segmented.tif")
			IJ.saveAs(classified_vis, "TIFF", segmented_path)
		
		# Use histogram to calculate area per class
		histogram = ImageStatistics.getStatistics(classified.getProcessor()).getHistogram()
		image_pixels = imp.getWidth()*imp.getHeight();
		
		# Create ParticleAnalyzer
		ParticleAnalyzer.setSummaryTable(classification_table)  # For some reason this is a static method for all particle analyzers
		particle_analyzer = ParticleAnalyzer(
			ParticleAnalyzer.SHOW_OUTLINES | ParticleAnalyzer.DISPLAY_SUMMARY,
			Measurements.AREA,
			None,
			min_particle_area, 1e100)
		particle_analyzer.setHideOutputImage(True)
		
		for i in range(num_classes):
			IJ.log("Running particle analyzer on class {}".format(class_labels[i]))
		
			# Select a single class by setting threshold min and max to this class value
			IJ.setRawThreshold(classified, i, i, "Red");
		
			# Count blobs using ParticleAnalyzer
			particle_analyzer.analyze(classified)

			if save_blob_images:
				# Particle analyzer visualization image
				particles_image = particle_analyzer.getOutputImage()
				particles_image.setTitle(image_title + " Blobs of class: " + class_labels[i])
	
				# Overlay particle image over original
				IJ.run(particles_image, "Invert LUT", "");
				IJ.run(particles_image, "Apply LUT", "");
				ic = ImageCalculator()
				overlay_image = ic.run("Transparent-zero create", imp, particles_image);
	
				# Show detected blobs
				if show_segmented:
					overlay_image.show()
	
				# Save detected blob images
				if segmented_folder is not None:
					blobs_path = os.path.join(segmented_folder.getAbsolutePath(), os.path.splitext(image_title)[0] + "-blobs-" + class_labels[i] + ".tif")
					IJ.saveAs(overlay_image, "TIFF", blobs_path)
				
			classification_table.addValue("Slice", class_labels[i])
		
			# The area calculated by the ParticleAnalyser excludes too small blobs
			# Additionally calculate the whole area with tiny blobs
			classification_table.addValue("File", image_title)
			classification_table.addValue("Complete Area", histogram[i])
			classification_table.addValue("Complete %Area", float(histogram[i])/image_pixels*100)
		
		
		# Update the table by calling show again
		classification_table.show("Classification Results")
	
	#Rename particle metrics to avoid confusion with complete area
	classification_table.renameColumn("Slice", "Class Label")
	classification_table.renameColumn("Total Area", "Particle Total Area")
	classification_table.renameColumn("Average Size", "Particle Average Size")
	classification_table.renameColumn("%Area", "Particle %Area")
	classification_table.renameColumn("Count", "Particle Count")
	
	# Update the table by calling show again
	classification_table.show("Classification Results")
	
	if result_file is not None:
		classification_table.saveAs(result_file.getAbsolutePath())
		# Append settings
		if append_settings:
			with open(result_file.getAbsolutePath(), "a") as result_file:
				result_file.write("\n")
				settings = {
					"Setting": "Value",
					"Classifier file": classifier_file,
					"Minimum blob area": min_particle_area
				}
				for k, v in settings.items():
					result_file.write("\"{}\",\"{}\"\n".format(k, v))
	
	IJ.showMessage("Classification batch done")

main()