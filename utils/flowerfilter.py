import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import os
from tqdm import tqdm
from flower_utils import object_detect, bbox_visualize

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


# Import Model
base_options = python.BaseOptions(model_asset_path='object detection/exported_model/flower_detection.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.4)
detector = vision.ObjectDetector.create_from_options(options)

# Flowers
# flowers = ['Achillea millefolium', 'Agalinis tenuifolia', 'Allium acuminatum', 'Allium cernuum', 'Ambrosia artemisiifolia', 'Ambrosia trifida', 'Amelanchier utahensis', 'Amorpha fruticosa', 'Amsinckia menziesii', 'Anaphalis margaritacea', 'Anemone virginiana', 'Anthemis cotula', 'Apios americana', 'Asclepias incarnata', 'Asclepias tuberosa', 'Astragalus canadensis', 'Astragalus schmolliae', 'Balsamorhiza sagittata', 'Bidens aristosa', 'Bidens bipinnata', 'Boechera laevigata', 'Calypso bulbosa', 'Calystegia sepium', 'Campanula rotundifolia', 'Campsis radicans', 'Carduus nutans', 'Castilleja linariifolia', 'Centaurea cyanus', 'Chaenactis douglasii', 'Chamerion angustifolium', 'Chrysanthemum leucanthemum', 'Cichorium intybus', 'Cirsium vulgare', 'Claytonia lanceolata', 'Claytonia perfoliata', 'Clematis hirsutissima', 'Clematis ligusticifolia', 'Clematis occidentalis', 'Cleomella palmeriana', 'Collinsia parviflora', 'Collomia grandiflora', 'Collomia linearis', 'Cordylanthus wrightii', 'Coreopsis lanceolata', 'Cornus canadensis', 'Cornus sericea', 'Coronilla varia', 'Crepis intermedia', 'Cuscuta gronovii', 'Cynoglossum officinale', 'Cypripedium parviflorum', 'Dasiphora fruticosa', 'Datura stramonium', 'Datura wrightii', 'Daucus carota', 'Delphinium carolinianum', 'Desmanthus illinoensis', 'Dianthus armeria', 'Echinacea purpurea', 'Echium vulgare', 'Equisetum laevigatum', 'Erigeron philadelphicus', 'Erigeron strigosus', 'Erodium cicutarium', 'Erythronium grandiflorum', 'Eschscholzia californica', 'Euphorbia marginata', 'Eutrochium maculatum', 'Fallopia japonica', 'Fragaria virginiana', 'Fritillaria pudica', 'Gaillardia pinnatifida', 'Gaillardia pulchella', 'Gentiana algida', 'Geranium viscosissimum', 'Glandularia canadensis', 'Glechoma hederacea', 'Grindelia squarrosa', 'Helenium autumnale', 'Helianthus annuus', 'Helianthus tuberosus', 'Hemerocallis fulva', 'Heracleum maximum', 'Hesperis matronalis', 'Heterotheca subaxillaris', 'Hieracium aurantiacum', 'Hydrophyllum capitatum', 'Hymenoxys grandiflora', 'Hypochaeris radicata', 'Hypoxis hirsuta', 'Impatiens capensis', 'Ipomopsis aggregata', 'Krigia biflora', 'Lactuca canadensis', 'Lactuca serriola', 'Lamium amplexicaule', 'Lamium purpureum', 'Lathyrus latifolius', 'Lilium philadelphicum', 'Linaria dalmatica', 'Linaria vulgaris', 'Linum lewisii', 'Lithophragma parviflorum', 'Lithospermum ruderale', 'Lobelia cardinalis', 'Lobelia siphilitica', 'Lotus corniculatus', 'Ludwigia alternifolia', 'Lupinus argenteus', 'Lycopus uniflorus', 'Lythrum salicaria', 'Mahonia repens', 'Maianthemum racemosum', 'Maianthemum stellatum', 'Matricaria discoidea', 'Medicago lupulina', 'Melampodium leucanthum', 'Melilotus albus', 'Melilotus officinalis', 'Mimulus nanus', 'Mitella stauropetala', 'Monarda fistulosa', 'Nuphar lutea', 'Nymphaea odorata', 'Oenothera pallida', 'Opuntia humifusa', 'Orobanche uniflora', 'Osmorhiza longistylis', 'Oxalis dillenii', 'Oxalis violacea', 'Pedicularis canadensis', 'Pedicularis centranthera', 'Peritoma serrulata', 'Persicaria amphibia', 'Persicaria punctata', 'Phacelia hastata', 'Phlox diffusa', 'Phyla lanceolata', 'Physalis virginiana', 'Plantago lanceolata', 'Polemonium pulcherrimum', 'Potentilla recta', 'Prunella vulgaris', 'Prunus americana', 'Prunus virginiana', 'Purshia stansburiana', 'Ranunculus abortivus', 'Ranunculus glaberrimus', 'Ratibida columnifera', 'Rhus aromatica', 'Rhus glabra', 'Rosa woodsii', 'Rubus parviflorus', 'Rudbeckia hirta', 'Rudbeckia laciniata', 'Rudbeckia triloba', 'Sagittaria latifolia', 'Sambucus nigra ssp canadensis', 'Sambucus nigra ssp cerulea', 'Sambucus racemosa', 'Sanguisorba minor', 'Saponaria officinalis', 'Silene latifolia', 'Solanum carolinense', 'Solanum dulcamara', 'Solidago altissima', 'Stanleya pinnata', 'Stellaria media', 'Streptanthus cordatus', 'Symphoricarpos albus', 'Symphyotrichum novae-angliae', 'Tanacetum vulgare', 'Taraxacum officinale', 'Teucrium canadense', 'Thalictrum occidentale', 'Thalictrum revolutum', 'Thermopsis montana', 'Toxicodendron rydbergii', 'Toxicoscordion paniculatum', 'Tragopogon dubius', 'Trautvetteria caroliniensis', 'Trifolium pratense', 'Trifolium repens', 'Trillium ovatum', 'Triodanis perfoliata', 'Triteleia grandiflora', 'Vaccaria hispanica', 'Verbascum blattaria', 'Verbascum thapsus', 'Veronica anagallis-aquatica', 'Veronica persica', 'Viola bicolor', 'Viola canadensis', 'Viola purpurea']
# start = 'Astragalus schmolliae'

flowers = []


# Source Directory
directory = 'images'

# Destination Directory
dataset_directory = 'dataset'

# Filtering Parameters
conf_threshold = 0.4
pixels = 200

# Iterate over each folder in source directory
for flower in tqdm(flowers):
    print(flower)
    oldpath = os.path.join(directory, flower.replace(' ', '_') + '_flower')
    if not os.path.exists(oldpath):
        print(f'Path not found: {flower}')
        continue
    newpath = os.path.join(dataset_directory, flower.replace(' ', '_'))
    if not os.path.exists(newpath) and newpath != 'dataset/images':
        os.makedirs(newpath)

    file_num = 0

    # For each file in directory
    for file in os.listdir(oldpath):
        if not file.startswith('.'):
            file_dir = os.path.join(oldpath, file)
            
            image = mp.Image.create_from_file(file_dir)
            im = Image.open(file_dir)
            im = im.convert('RGB')
            
            # Add image directly to dataset if image size is small (500 x 500)
            if im.size[0] < 500 and im.size[1] < 500:
                new_file = flower.replace(' ', '_') + str(file_num)
                im = im.resize((300,300))
                im.save(os.path.join(newpath, new_file)+'.jpg')
                file_num += 1
                continue

            detection_result = detector.detect(image)

            # Iterate through the objects identified by the model
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

                new_file = flower.replace(' ', '_') + str(file_num)
                im1 = im.crop((bbox.origin_x, bbox.origin_y, bbox.origin_x + bbox.width, bbox.origin_y + bbox.height))
                im1 = im1.resize((300,300))
                im1.save(os.path.join(newpath, new_file)+'.jpg')
                file_num += 1

                

                
            
        
        