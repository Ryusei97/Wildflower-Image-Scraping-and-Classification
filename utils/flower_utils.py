# Custom functions for detection and visualization

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
import os
import tensorflow as tf
import matplotlib.pyplot as plt


CLASS_NAMES = ['Achillea_millefolium', 'Agalinis_tenuifolia', 'Allium_acuminatum', 'Allium_cernuum', 'Amelanchier_utahensis', 'Amorpha_fruticosa', 'Amsinckia_menziesii', 'Anaphalis_margaritacea', 'Anemone_virginiana', 'Anthemis_cotula', 'Apios_americana', 'Asclepias_incarnata', 'Asclepias_tuberosa', 'Balsamorhiza_sagittata', 'Bidens_aristosa', 'Calypso_bulbosa', 'Calystegia_sepium', 'Campanula_rotundifolia', 'Campsis_radicans', 'Carduus_nutans', 'Castilleja_linariifolia', 'Centaurea_cyanus', 'Chaenactis_douglasii', 'Chamerion_angustifolium', 'Chrysanthemum_leucanthemum', 'Cichorium_intybus', 'Cirsium_vulgare', 'Claytonia_lanceolata', 'Claytonia_perfoliata', 'Clematis_hirsutissima', 'Clematis_ligusticifolia', 'Clematis_occidentalis', 'Collinsia_parviflora', 'Collomia_grandiflora', 'Coreopsis_lanceolata', 'Cornus_canadensis', 'Cornus_sericea', 'Coronilla_varia', 'Cynoglossum_officinale', 'Cypripedium_parviflorum', 'Dasiphora_fruticosa', 'Datura_stramonium', 'Datura_wrightii', 'Daucus_carota', 'Delphinium_carolinianum', 'Dianthus_armeria', 'Echinacea_purpurea', 'Echium_vulgare', 'Erigeron_philadelphicus', 'Erigeron_strigosus', 'Erodium_cicutarium', 'Erythronium_grandiflorum', 'Eschscholzia_californica', 'Euphorbia_marginata', 'Eutrochium_maculatum', 'Fallopia_japonica', 'Fragaria_virginiana', 'Fritillaria_pudica', 'Gaillardia_pinnatifida', 'Gaillardia_pulchella', 'Gentiana_algida', 'Geranium_viscosissimum', 'Glandularia_canadensis', 'Glechoma_hederacea', 'Grindelia_squarrosa', 'Helenium_autumnale', 'Helianthus_annuus', 'Helianthus_tuberosus', 'Hemerocallis_fulva', 'Heracleum_maximum', 'Hesperis_matronalis', 'Heterotheca_subaxillaris', 'Hieracium_aurantiacum', 'Hydrophyllum_capitatum', 'Hymenoxys_grandiflora', 'Hypochaeris_radicata', 'Hypoxis_hirsuta', 'Impatiens_capensis', 'Ipomopsis_aggregata', 'Krigia_biflora', 'Lactuca_serriola', 'Lamium_amplexicaule', 'Lamium_purpureum', 'Lathyrus_latifolius', 'Lilium_philadelphicum', 'Linaria_vulgaris', 'Linum_lewisii', 'Lithophragma_parviflorum', 'Lobelia_cardinalis', 'Lobelia_siphilitica', 'Lotus_corniculatus', 'Ludwigia_alternifolia', 'Lupinus_argenteus', 'Lythrum_salicaria', 'Mahonia_repens', 'Maianthemum_racemosum', 'Maianthemum_stellatum', 'Matricaria_discoidea', 'Medicago_lupulina', 'Melampodium_leucanthum', 'Melilotus_albus', 'Melilotus_officinalis', 'Mimulus_nanus', 'Monarda_fistulosa', 'Nuphar_lutea', 'Nymphaea_odorata', 'Oenothera_pallida', 'Opuntia_humifusa', 'Orobanche_uniflora', 'Osmorhiza_longistylis', 'Oxalis_dillenii', 'Oxalis_violacea', 'Pedicularis_canadensis', 'Persicaria_amphibia', 'Phacelia_hastata', 'Phlox_diffusa', 'Plantago_lanceolata', 'Polemonium_pulcherrimum', 'Potentilla_recta', 'Prunella_vulgaris', 'Prunus_americana', 'Prunus_virginiana', 'Purshia_stansburiana', 'Ranunculus_abortivus', 'Ranunculus_glaberrimus', 'Ratibida_columnifera', 'Rhus_aromatica', 'Rosa_woodsii', 'Rubus_parviflorus', 'Rudbeckia_hirta', 'Rudbeckia_laciniata', 'Rudbeckia_triloba', 'Sagittaria_latifolia', 'Sambucus_racemosa', 'Sanguisorba_minor', 'Saponaria_officinalis', 'Silene_latifolia', 'Solanum_carolinense', 'Solanum_dulcamara', 'Solidago_altissima', 'Stanleya_pinnata', 'Stellaria_media', 'Symphoricarpos_albus', 'Symphyotrichum_novae-angliae', 'Tanacetum_vulgare', 'Taraxacum_officinale', 'Teucrium_canadense', 'Thalictrum_occidentale', 'Thermopsis_montana', 'Tragopogon_dubius', 'Trautvetteria_caroliniensis', 'Trifolium_pratense', 'Trifolium_repens', 'Trillium_ovatum', 'Triodanis_perfoliata', 'Triteleia_grandiflora', 'Vaccaria_hispanica', 'Verbascum_blattaria', 'Verbascum_thapsus', 'Veronica_anagallis-aquatica', 'Veronica_persica', 'Viola_bicolor', 'Viola_canadensis', 'Viola_purpurea']


# Function to visualize predictions with bounding boxes and labels
def visualize_predictions(image_path, detector, image_classifier_model, output_path=None):
    # Import the bounding box model
    
    # Import the classifier model
    

    # Classnames and thumbnails
    thumb_dir = 'thumbnails'  

    # Read and resize the image using PIL
    pil_image = Image.open(image_path)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_image))

    # Predict bounding boxes using the pil_image and the bounding box model (replace with your own implementation)
    detection_result = detector.detect(image).detections

    # Create a blank padded image
    padded_image = Image.new('RGB', (1080, 720))

    # Paste the resized image onto the padded image
    padded_image.paste(pil_image.resize((720, 720)), (0, 0))

    # Draw bounding boxes and labels
    for i, det in enumerate(detection_result):
        box = det.bounding_box
        bounding_box = (box.origin_x, box.origin_y, box.width, box.height)  # Replace with your bounding box coordinates

        new_width = 720  # Replace with your desired width
        new_height = 720  # Replace with your desired height

        scale_x = new_width / image.width
        scale_y = new_height / image.height

        resized_bounding_box = (
            int(bounding_box[0] * scale_x),
            int(bounding_box[1] * scale_y),
            int(bounding_box[2] * scale_x),
            int(bounding_box[3] * scale_y)
        )

        # Draw red bounding box
        draw = ImageDraw.Draw(padded_image)
        draw.rectangle([resized_bounding_box[0], resized_bounding_box[1], resized_bounding_box[0] + resized_bounding_box[2], resized_bounding_box[1] + resized_bounding_box[3]], outline='red', width=3)
        draw.rectangle([resized_bounding_box[0], resized_bounding_box[1], resized_bounding_box[0] + 25, resized_bounding_box[1] + 35], fill='red')


        # Label the bounding box with a number
        label = str(i + 1)
        draw.text((resized_bounding_box[0] + 5, resized_bounding_box[1] + 5), label, fill='white', font=ImageFont.truetype("utils/arial/arial.ttf", 25))

        # Crop the bounding box region from the image
        cropped_image = pil_image.crop((box.origin_x, box.origin_y, box.origin_x + box.width, box.origin_y + box.height))
        cropped_image = cropped_image.resize((224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(cropped_image)
        
        # Add batch dimension to the image
        preprocessed_image = tf.expand_dims(image_array, axis=0)
        
        # Get the prediction results from the image classifier model (replace with your own implementation)
        prediction = image_classifier_model.predict(preprocessed_image, verbose=0)
        
        top_classes = tf.argsort(prediction, axis=1, direction='DESCENDING')[0, :3]
        # Get the top 3 predicted class indices and their corresponding probabilities

        top_probs = prediction[0][top_classes]

        # Create the label string with top class names and probabilities
        label_str = '\n'.join([f'{CLASS_NAMES[idx]}: {prob*100:.2f}%' for idx, prob in zip(top_classes, top_probs)])

        # Add the bounding box number and top predictions to the padded image
        draw.text((770, (i * 160) + 80), f'Box {i+1}:\n{label_str}', fill='red', font=ImageFont.truetype("utils/arial/arial.ttf", size=18))

        # Add the thumbnails
        for j, idx in enumerate(top_classes):
            path = os.path.join('thumbnails', CLASS_NAMES[idx])
            thumb_path = os.path.join(path, os.listdir(path)[0])
            thumbnail = Image.open(thumb_path).resize((70,50))
            padded_image.paste(thumbnail, (780+j*(70 + 10), (i * 160) + 180))

        if i == 3:
            break


    # Display the final image with bounding boxes and labels
    plt.imshow(padded_image)
    plt.axis('off')

    # Save the output image if output_path is provided
    if output_path:
        padded_image.save(output_path)

    plt.show()

