# Experiment 1: examples (verifier v2)

Rows where NLLB-600M fails a check, shown next to Google, Llama-3.1-8B and IGSC-repaired Llama. Cells marked ✗ fail the same check.

## broken_placeholder

| lang | line | source | Google (NMT) | NLLB-600M (NMT) | Llama-3.1-8B (LLM) | Llama-3.1-8B (LLM+IGSC) |
|---|---|---|---|---|---|---|
| hi | 333 | `%1 unsaved document (%2)` | `%1 असहेजित दस्तावेज़ (%2)` | `% 1 सहेजा नहीं गया दस्तावेज़ (% 2)` ✗ | `%1 अनसेव दस्तावेज़ (%2)` | `%1 अनसेव दस्तावेज़ (%2)` |
| hi | 490 | `Network interface %1 disconnected` | `नेटवर्क इंटरफ़ेस %1 डिस्कनेक्ट हो गया` | `नेटवर्क इंटरफ़ेस % 1 डिस्कनेक्ट` ✗ | `नेटवर्क इंटरफ़ेस %1 बंद हो गया है` | `नेटवर्क इंटरफ़ेस %1 बंद हो गया है` |
| hi | 513 | `Copy %1 to:` | `%1 को कॉपी करें:` | `% 1 को कॉपी करेंः` ✗ | `%1 को:` | `%1 को:` |
| hi | 682 | `& Location of %1` | `& %1 का स्थान` | `% 1 का स्थान` ✗ | `&amp; स्थान %1 का` | `&amp; स्थान %1 का` |
| hi | 857 | `Save as %1` | `1% के रूप में सहेजें` ✗ | `% 1 के रूप में सहेजें` ✗ | `%1 के रूप में सेवा करें` | `%1 के रूप में सेवा करें` |
| hi | 892 | `Help & with %1` | `सहायता और %1 के साथ` | `% 1 के साथ सहायता` ✗ | `मदद (&वि) के साथ %1` | `मदद (&वि) के साथ %1` |

## broken_accelerator

| lang | line | source | Google (NMT) | NLLB-600M (NMT) | Llama-3.1-8B (LLM) | Llama-3.1-8B (LLM+IGSC) |
|---|---|---|---|---|---|---|
| hi | 23 | `of the & year` | `साल का` ✗ | `वर्ष का` ✗ | `& वर्ष के` | `& वर्ष के` |
| hi | 95 | `& Folder contents:` | `फ़ोल्डर की सामग्री:` ✗ | `फ़ोल्डर सामग्रीः` ✗ | `&amp; फ़ोल्डर के सामग्री:` ✗ | `&amp; फ़ोल्डर के सामग्री:` ✗ |
| hi | 234 | `Template & name:` | `टेम्पलेट और नाम:` ✗ | `टेम्पलेट और नामः` ✗ | `सहायक और नाम:` ✗ | `सहायक और नाम:` ✗ |
| hi | 237 | `Acceleration & profile:` | `त्वरण और प्रोफ़ाइल:` ✗ | `त्वरण और प्रोफ़ाइलः` ✗ | `तीव्रता और प्रोफ़ाइल:` ✗ | `तीव्रता और प्रोफ़ाइल:` ✗ |
| hi | 271 | `Last & Friday` | `अंतिम और शुक्रवार` ✗ | `पिछले और शुक्रवार` ✗ | `अंतिम और शुक्रवार` ✗ | `अंतिम और शुक्रवार` ✗ |
| hi | 509 | `Archive & file:` | `संग्रह फ़ाइल:` ✗ | `संग्रह और फ़ाइलः` ✗ | `आर्काइव और फ़ाइल:` ✗ | `आर्काइव और फ़ाइल:` ✗ |

## copy_fallback

| lang | line | source | Google (NMT) | NLLB-600M (NMT) | Llama-3.1-8B (LLM) | Llama-3.1-8B (LLM+IGSC) |
|---|---|---|---|---|---|---|
| hi | 106 | `ImagePlugin_SuperImpose` | `ImagePlugin_SuperImpose` ✗ | `ImagePlugin_SuperImpose` ✗ | `सुपर इम्पोज़ प्लगइन` | `सुपर इम्पोज़ प्लगइन` |
| hi | 127 | `KDevStructureTree` | `KDevStructureTree` ✗ | `KDevStructureTree` ✗ | `कडेव संरचना पेड़` | `कडेव संरचना पेड़` |
| hi | 325 | `CGConstraintSolver` | `सीजी कंस्ट्रेंट सॉल्वर` | `CGConstraintSolver` ✗ | `CG संगति समाधानकर्ता` | `CG संगति समाधानकर्ता` |
| hi | 569 | `ImagePlugin_Border` | `ImagePlugin_Border` ✗ | `ImagePlugin_Border` ✗ | `बॉर्डर इमेज प्लगइन` | `बॉर्डर इमेज प्लगइन` |
| hi | 884 | `new_component` | `नया_घटक` | `new_component` ✗ | `नया_सदस्य` | `नया_सदस्य` |
| hi | 1066 | `rectParticleCount` | `रेक्टपार्टिकलकाउंट` | `rectParticleCount` ✗ | `पार्श्व कणों की संख्या` | `पार्श्व कणों की संख्या` |
