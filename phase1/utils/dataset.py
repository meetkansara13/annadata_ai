"""
अन्नदाता AI — Phase 1
Dataset utility — with full multilingual labels for all 15 classes.
"""

from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

DATA_DIR    = Path("data/PlantVillage/PlantVillage")
IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_WORKERS = 0
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.10
SEED        = 42

CLASS_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "en": "Bell Pepper Bacterial Spot",
        "hi": "शिमला मिर्च का बैक्टीरियल धब्बा",
        "gu": "શિમલા મરચાનો બેક્ટેરિયલ ડાઘ",
        "mr": "ढोबळी मिरचीचा जिवाणू डाग",
        "pa": "ਸ਼ਿਮਲਾ ਮਿਰਚ ਦਾ ਬੈਕਟੀਰੀਅਲ ਧੱਬਾ",
        "ta": "குடைமிளகாய் பாக்டீரியல் புள்ளி",
        "te": "బెల్ పెప్పర్ బాక్టీరియల్ మచ్చ",
        "bn": "ক্যাপসিকামের ব্যাকটেরিয়াল দাগ",
        "kn": "ಕ್ಯಾಪ್ಸಿಕಂ ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಚುಕ್ಕೆ",
        "ml": "ബെൽ പെപ്പർ ബാക്ടീരിയൽ പുള്ളി",
    },
    "Pepper__bell___healthy": {
        "en": "Bell Pepper — Healthy",
        "hi": "शिमला मिर्च — स्वस्थ",
        "gu": "શિમલા મરચું — સ્વસ્થ",
        "mr": "ढोबळी मिरची — निरोगी",
        "pa": "ਸ਼ਿਮਲਾ ਮਿਰਚ — ਸਿਹਤਮੰਦ",
        "ta": "குடைமிளகாய் — ஆரோக்கியமானது",
        "te": "బెల్ పెప్పర్ — ఆరోగ్యకరమైనది",
        "bn": "ক্যাপসিকাম — সুস্থ",
        "kn": "ಕ್ಯಾಪ್ಸಿಕಂ — ಆರೋಗ್ಯಕರ",
        "ml": "ബെൽ പെപ്പർ — ആരോഗ്യകരം",
    },
    "Potato___Early_blight": {
        "en": "Potato Early Blight",
        "hi": "आलू का अगेती झुलसा",
        "gu": "બટાકાનો વહેલો સુકારો",
        "mr": "बटाट्याचा अगेती करपा",
        "pa": "ਆਲੂ ਦੀ ਅਗੇਤੀ ਝੁਲਸ",
        "ta": "உருளைக்கிழங்கின் ஆரம்பகால கருகல்",
        "te": "బంగాళాదుంప ముందస్తు తెగులు",
        "bn": "আলুর আগাম ধ্বসা",
        "kn": "ಆಲೂಗಡ್ಡೆ ಮೊದಲ ಸುಟ್ಟ ರೋಗ",
        "ml": "ഉരുളക്കിഴങ്ങ് ആദ്യകാല കരിച്ചിൽ",
    },
    "Potato___Late_blight": {
        "en": "Potato Late Blight",
        "hi": "आलू का पिछेती झुलसा",
        "gu": "બટાકાનો મોડો સુકારો",
        "mr": "बटाट्याचा पिछेती करपा",
        "pa": "ਆਲੂ ਦੀ ਪਿਛੇਤੀ ਝੁਲਸ",
        "ta": "உருளைக்கிழங்கின் தாமதமான கருகல்",
        "te": "బంగాళాదుంప ఆలస్య తెగులు",
        "bn": "আলুর পিছেতা ধ্বসা",
        "kn": "ಆಲೂಗಡ್ಡೆ ತಡ ಸುಟ್ಟ ರೋಗ",
        "ml": "ഉരുളക്കിഴങ്ങ് വൈകിയ കരിച്ചിൽ",
    },
    "Potato___healthy": {
        "en": "Potato — Healthy",
        "hi": "आलू — स्वस्थ",
        "gu": "બટાકા — સ્વસ્થ",
        "mr": "बटाटा — निरोगी",
        "pa": "ਆਲੂ — ਸਿਹਤਮੰਦ",
        "ta": "உருளைக்கிழங்கு — ஆரோக்கியமானது",
        "te": "బంగాళాదుంప — ఆరోగ్యకరమైనది",
        "bn": "আলু — সুস্থ",
        "kn": "ಆಲೂಗಡ್ಡೆ — ಆರೋಗ್ಯಕರ",
        "ml": "ഉരുളക്കിഴങ്ങ് — ആരോഗ്യകരം",
    },
    "Tomato_Bacterial_spot": {
        "en": "Tomato Bacterial Spot",
        "hi": "टमाटर का बैक्टीरियल धब्बा",
        "gu": "ટામેટાનો બેક્ટેરિયલ ડાઘ",
        "mr": "टोमॅटोचा जिवाणू डाग",
        "pa": "ਟਮਾਟਰ ਦਾ ਬੈਕਟੀਰੀਅਲ ਧੱਬਾ",
        "ta": "தக்காளி பாக்டீரியல் புள்ளி",
        "te": "టమాటా బాక్టీరియల్ మచ్చ",
        "bn": "টমেটোর ব্যাকটেরিয়াল দাগ",
        "kn": "ಟೊಮೇಟೊ ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಚುಕ್ಕೆ",
        "ml": "തക്കാളി ബാക്ടീരിയൽ പുള്ളി",
    },
    "Tomato_Early_blight": {
        "en": "Tomato Early Blight",
        "hi": "टमाटर का अगेती झुलसा",
        "gu": "ટામેટાનો વહેલો સુકારો",
        "mr": "टोमॅटोचा अगेती करपा",
        "pa": "ਟਮਾਟਰ ਦੀ ਅਗੇਤੀ ਝੁਲਸ",
        "ta": "தக்காளியின் ஆரம்பகால கருகல்",
        "te": "టమాటా ముందస్తు తెగులు",
        "bn": "টমেটোর আগাম ধ্বসা",
        "kn": "ಟೊಮೇಟೊ ಮೊದಲ ಸುಟ್ಟ ರೋಗ",
        "ml": "തക്കാളി ആദ്യകാല കരിച്ചിൽ",
    },
    "Tomato_Late_blight": {
        "en": "Tomato Late Blight",
        "hi": "टमाटर का पिछेती झुलसा",
        "gu": "ટામેટાનો મોડો સુકારો",
        "mr": "टोमॅटोचा पिछेती करपा",
        "pa": "ਟਮਾਟਰ ਦੀ ਪਿਛੇਤੀ ਝੁਲਸ",
        "ta": "தக்காளியின் தாமதமான கருகல்",
        "te": "టమాటా ఆలస్య తెగులు",
        "bn": "টমেটোর পিছেতা ধ্বসা",
        "kn": "ಟೊಮೇಟೊ ತಡ ಸುಟ್ಟ ರೋಗ",
        "ml": "തക്കാളി വൈകിയ കരിച്ചിൽ",
    },
    "Tomato_Leaf_Mold": {
        "en": "Tomato Leaf Mold",
        "hi": "टमाटर का फफूंद",
        "gu": "ટામેટાની પાનની ફૂગ",
        "mr": "टोमॅटोची पानांची बुरशी",
        "pa": "ਟਮਾਟਰ ਦੀ ਪੱਤੀ ਦੀ ਉੱਲੀ",
        "ta": "தக்காளி இலை அச்சு",
        "te": "టమాటా ఆకు అచ్చు",
        "bn": "টমেটোর পাতার ছত্রাক",
        "kn": "ಟೊಮೇಟೊ ಎಲೆ ಅಚ್ಚು",
        "ml": "തക്കാളി ഇല പൂപ്പൽ",
    },
    "Tomato_Septoria_leaf_spot": {
        "en": "Tomato Septoria Leaf Spot",
        "hi": "टमाटर का सेप्टोरिया धब्बा",
        "gu": "ટામેટાનો સેપ્ટોરિયા પાન ડાઘ",
        "mr": "टोमॅटोचा सेप्टोरिया पान डाग",
        "pa": "ਟਮਾਟਰ ਦਾ ਸੇਪਟੋਰੀਆ ਧੱਬਾ",
        "ta": "தக்காளி செப்டோரியா இலை புள்ளி",
        "te": "టమాటా సెప్టోరియా ఆకు మచ్చ",
        "bn": "টমেটোর সেপ্টোরিয়া দাগ",
        "kn": "ಟೊಮೇಟೊ ಸೆಪ್ಟೋರಿಯಾ ಎಲೆ ಚುಕ್ಕೆ",
        "ml": "തക്കാളി സെപ്റ്റോറിയ ഇല പുള്ളി",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "en": "Tomato Spider Mites",
        "hi": "टमाटर का मकड़ी कीट",
        "gu": "ટામેટાનો કરોળિયો જીવાત",
        "mr": "टोमॅटोचा कोळी कीड",
        "pa": "ਟਮਾਟਰ ਦੀ ਮੱਕੜੀ ਕੀਟ",
        "ta": "தக்காளி சிலந்தி பூச்சி",
        "te": "టమాటా సాలె పురుగు",
        "bn": "টমেটোর মাকড়সা পোকা",
        "kn": "ಟೊಮೇಟೊ ಜೇಡ ಹುಳ",
        "ml": "തക്കാളി ചിലന്തി പ്രാണി",
    },
    "Tomato__Target_Spot": {
        "en": "Tomato Target Spot",
        "hi": "टमाटर का लक्ष्य धब्बा",
        "gu": "ટામેટાનો ટાર્ગેટ ડાઘ",
        "mr": "टोमॅटोचा लक्ष्य डाग",
        "pa": "ਟਮਾਟਰ ਦਾ ਟਾਰਗੇਟ ਧੱਬਾ",
        "ta": "தக்காளி இலக்கு புள்ளி",
        "te": "టమాటా లక్ష్య మచ్చ",
        "bn": "টমেটোর লক্ষ্য দাগ",
        "kn": "ಟೊಮೇಟೊ ಗುರಿ ಚುಕ್ಕೆ",
        "ml": "തക്കാളി ടാർഗറ്റ് പുള്ളി",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "en": "Tomato Yellow Leaf Curl Virus",
        "hi": "टमाटर का पीला पत्ता मरोड़ वायरस",
        "gu": "ટામેટાનો પીળો પાન વળ વાઈરસ",
        "mr": "टोमॅटोचा पिवळा पान मुरगळ विषाणू",
        "pa": "ਟਮਾਟਰ ਦਾ ਪੀਲਾ ਪੱਤਾ ਮਰੋੜ ਵਾਇਰਸ",
        "ta": "தக்காளி மஞ்சள் இலை சுருள் வைரஸ்",
        "te": "టమాటా పసుపు ఆకు మురి వైరస్",
        "bn": "টমেটোর হলুদ পাতা কুঁকড়ানো ভাইরাস",
        "kn": "ಟೊಮೇಟೊ ಹಳದಿ ಎಲೆ ಸುರಳಿ ವೈರಸ್",
        "ml": "തക്കാളി മഞ്ഞ ഇല ചുരുൾ വൈറസ്",
    },
    "Tomato__Tomato_mosaic_virus": {
        "en": "Tomato Mosaic Virus",
        "hi": "टमाटर का मोज़ेक वायरस",
        "gu": "ટામેટાનો મોઝેક વાઈરસ",
        "mr": "टोमॅटोचा मोझेक विषाणू",
        "pa": "ਟਮਾਟਰ ਦਾ ਮੋਜ਼ੇਕ ਵਾਇਰਸ",
        "ta": "தக்காளி மொசைக் வைரஸ்",
        "te": "టమాటా మొజాయిక్ వైరస్",
        "bn": "টমেটোর মোজাইক ভাইরাস",
        "kn": "ಟೊಮೇಟೊ ಮೊಸಾಯಿಕ್ ವೈರಸ್",
        "ml": "തക്കാളി മൊസൈക് വൈറസ്",
    },
    "Tomato_healthy": {
        "en": "Tomato — Healthy",
        "hi": "टमाटर — स्वस्थ",
        "gu": "ટામેટું — સ્વસ્થ",
        "mr": "टोमॅटो — निरोगी",
        "pa": "ਟਮਾਟਰ — ਸਿਹਤਮੰਦ",
        "ta": "தக்காளி — ஆரோக்கியமானது",
        "te": "టమాటా — ఆరోగ్యకరమైనది",
        "bn": "টমেটো — সুস্থ",
        "kn": "ಟೊಮೇಟೊ — ಆರೋಗ್ಯಕರ",
        "ml": "തക്കാളി — ആരോഗ്യകരം",
    },
}

TREATMENT_ADVICE = {
    "Potato___Early_blight":    "Apply Mancozeb 75% WP @ 2g/L water. Remove infected leaves. Spray every 10 days.",
    "Potato___Late_blight":     "Apply Metalaxyl + Mancozeb immediately. Avoid overhead irrigation. Destroy infected plants.",
    "Tomato_Early_blight":      "Spray Copper Oxychloride 50% WP @ 3g/L. Maintain plant spacing for airflow.",
    "Tomato_Late_blight":       "Apply Cymoxanil + Mancozeb immediately. Remove and burn infected leaves.",
    "Tomato_Bacterial_spot":    "Copper bactericide @ 3g/L. Avoid wet-field work. Use disease-free seeds.",
    "Tomato_Leaf_Mold":         "Improve ventilation. Apply Chlorothalonil @ 2g/L. Reduce humidity.",
    "Tomato_Septoria_leaf_spot":"Remove lower infected leaves. Apply Mancozeb or Copper fungicide.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray Abamectin @ 0.5ml/L or neem oil @ 5ml/L.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "No cure — remove infected plants. Control whitefly with yellow sticky traps.",
    "Pepper__bell___Bacterial_spot": "Copper-based bactericide spray. Use disease-free seeds. Avoid overhead irrigation.",
    "Tomato__Target_Spot":      "Apply Chlorothalonil or Mancozeb. Remove infected leaves. Improve air circulation.",
    "Tomato__Tomato_mosaic_virus": "No cure. Remove infected plants. Wash hands and tools before touching healthy plants.",
}

def get_transforms(augment=True):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE+20, IMG_SIZE+20)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_dataloaders(data_dir=DATA_DIR):
    if not data_dir.exists():
        raise FileNotFoundError(f"\n[अन्नदाता AI] Dataset not found at '{data_dir}'.\n")
    full = datasets.ImageFolder(root=str(data_dir))
    n = len(full)
    n_test  = int(n * TEST_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    n_train = n - n_val - n_test
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=g)
    train_ds.dataset.transform = get_transforms(augment=True)
    val_ds.dataset.transform   = get_transforms(augment=False)
    test_ds.dataset.transform  = get_transforms(augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    print(f"\n[अन्नदाता AI] Dataset loaded ✓  Classes={len(full.classes)}  Train={n_train}  Val={n_val}  Test={n_test}")
    return train_loader, val_loader, test_loader, full.classes