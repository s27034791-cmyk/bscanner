# Business Card OCR Scanner

A sophisticated web-based business card scanner that uses advanced computer vision and AI to extract contact information from business card images. This application combines OpenCV for image preprocessing with Groq's Llama-4 vision model for intelligent text extraction, with special focus on Indian name detection.

## Features

- **Advanced Image Preprocessing**: Multiple preprocessing variants using OpenCV
- **AI-Powered OCR**: Utilizes Groq's Llama-4-scout-17b-16e-instruct model for vision-based text extraction
- **Indian Name Detection**: Specialized prompts and techniques for identifying Indian names regardless of font styling
- **Zone Detection**: Intelligent card layout analysis to identify text regions and spatial relationships
- **QR Code Decoding**: Built-in QR code detection and extraction
- **Multiple Format Support**: Handles various image formats and orientations
- **Web Interface**: Clean, responsive web UI for easy business card scanning
- **Knowledge Base**: Stores confirmed extractions for future reference

## Project Structure

```
vision_openai/
├── app.py                 # Main Flask application with image processing and OCR
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (API keys)
├── .gitignore           # Git ignore rules
├── templates/
│   └── index.html       # Main web interface
├── static/
│   ├── style.css        # CSS styling
│   └── app.js           # Frontend JavaScript
└── knowledgebase.json   # Storage for confirmed extractions
```

## Libraries Used

### Core Dependencies
- **Flask** (2.x): Web framework for the application server
- **python-dotenv**: Environment variable management
- **opencv-python**: Computer vision library for image processing
- **numpy**: Numerical computing for image operations
- **requests**: HTTP client for Groq API calls

### System Dependencies
- **Python 3.8+**: Required runtime environment

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Groq API key

### Step 1: Clone or Download the Project
```bash
cd vision_openai
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

To get a Groq API key:
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Generate a new API key
5. Copy and paste it into your `.env` file

## How to Use

### Starting the Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

### Using the Web Interface
1. Open your browser and navigate to `http://localhost:5000`
2. Click "Upload Image" to select a business card image from your device
3. Or click "Take Picture" to use your device's camera (mobile devices)
4. Click "Scan Card" to process the image
5. Review the extracted information in the form fields
6. Edit any fields if necessary
7. Click "Confirm Entries" to save the data to the knowledge base

### API Usage
You can also use the API directly:

**POST /scan**
- Upload a business card image
- Returns extracted contact information in JSON format

Example:
```bash
curl -X POST -F "image=@card.jpg" http://localhost:5000/scan
```

## What Has Been Done

### Image Processing Pipeline
The application implements a sophisticated image preprocessing pipeline:

1. **Image Loading and Normalization**
   - Automatic image decoding and validation
   - Resolution optimization (900px minimum, 2400px maximum)
   - High-quality resizing using Lanczos interpolation

2. **Quality Assessment**
   - Blur detection using Laplacian variance
   - Darkness detection using mean pixel intensity
   - Automatic adjustment based on image quality

3. **Image Enhancement**
   - **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Improves local contrast
   - **Deskewing**: Corrects tilt up to ±10° using Hough line detection
   - **Denoising**: Fast Non-local Means Denoising for noise reduction
   - **Deblurring**: Richardson-Lucy deconvolution with multi-scale unsharp masking
   - **Gamma Correction**: Adjusts brightness for dark images
   - **Bilateral Filtering**: Edge-preserving smoothing

4. **Preprocessing Variants**
   The system generates 4 complementary preprocessing variants:
   - **V1**: Bilateral-smoothed base for clean cards
   - **V2**: Deblur or hard sharpen based on blur score
   - **V3**: Adaptive threshold for gradient/patterned backgrounds
   - **V4**: Gamma-brightened CLAHE or morphological top-hat

5. **Zone Detection**
   - Contour analysis to identify text blocks
   - Icon anchor detection for contact field hints
   - Large vs small text classification
   - QR code detection and decoding
   - Spatial relationship mapping

### OCR and Text Extraction
1. **Groq Vision Integration**
   - Sends preprocessed images to Groq's Llama-4-scout model
   - Provides structural hints from zone detection
   - Includes QR data as additional context
   - Temperature set to 0.0 for consistent results

2. **Indian Name Detection**
   - Specialized prompts for Indian name recognition
   - Visual hierarchy analysis (font size, prominence, position)
   - Multi-line name reassembly
   - OCR artifact correction (e.g., "JO HN" → "JOHN")
   - Support for various Indian name formats

3. **Field Extraction**
   The system extracts the following fields:
   - **Name**: Person's full name (with special Indian name handling)
   - **Number**: Phone/WhatsApp/mobile numbers
   - **Email**: Email addresses with validation
   - **Address**: Full postal address with Indian format support
   - **Website**: Company website URLs
   - **Company Name**: Organization/business name
   - **Designation**: Job title or role

4. **Fallback Mechanism**
   - Automatic rotation retry if initial extraction fails
   - Name-only extraction prompt when full extraction returns null
   - Visual hierarchy-based name identification

### Web Interface
- Responsive design for desktop and mobile
- Image upload and camera capture support
- Real-time scanning progress display
- Editable form fields for extracted data
- Knowledge base integration for storing confirmed data

## Specialized Features

### Indian Name Detection
The application includes advanced techniques for detecting Indian names:

1. **Visual Hierarchy Analysis**
   - Identifies names as most prominent text
   - Considers font size, position, and styling
   - Handles all-caps, unique fonts, and artistic styling

2. **Multi-Format Support**
   - Handles split names across multiple lines
   - Recognizes common Indian name patterns
   - Supports names that look like company logos
   - Handles names with different fonts, sizes, or colors

3. **OCR Artifact Correction**
   - Fixes spacing issues: "JO HN" → "JOHN"
   - Corrects character confusion: "0↔O", "1↔l↔I"
   - Handles special character encoding issues

### Indian Address Handling
- Recognizes Indian address formats
- Identifies common Indian address indicators
- Handles multi-line addresses
- Supports landmark-based addresses

## Tesseract OCR (Optional Enhancement)

While this project primarily uses Groq's vision model for OCR, you can enhance it with Tesseract for additional capabilities.

### Installing Tesseract

#### Windows
1. Download Tesseract from the official GitHub repository:
   ```
   https://github.com/UB-Mannheim/tesseract/wiki
   ```
2. Run the installer (choose your language during installation)
3. Add Tesseract to your system PATH:
   - Default installation path: `C:\Program Files\Tesseract-OCR`
   - Add to PATH: `C:\Program Files\Tesseract-OCR`

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-eng  # English language pack
```

#### macOS
```bash
brew install tesseract
brew install tesseract-lang  # Additional language packs
```

### Installing Python Tesseract Wrapper
```bash
pip install pytesseract
```

### Using Tesseract in the Project
To integrate Tesseract, you would:

1. Add pytesseract to requirements.txt:
   ```
   pytesseract==0.3.10
   ```

2. Import and use in app.py:
   ```python
   import pytesseract
   
   # Set Tesseract path (Windows only)
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   
   # Extract text from image
   text = pytesseract.image_to_string(image)
   ```

3. For better results with business cards, configure Tesseract:
   ```python
   config = '--psm 6 --oem 3'  # Page segmentation mode and OCR engine mode
   text = pytesseract.image_to_string(image, config=config)
   ```

### Tesseract vs. Groq Vision
- **Tesseract**: Open-source, runs locally, supports many languages, but requires configuration
- **Groq Vision**: AI-powered, better at understanding context, handles artistic fonts better, requires API

## API Reference

### POST /scan
Upload and scan a business card image.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
  "raw": {
    "name": "John Doe",
    "number": "+1 555-123-4567",
    "email": "john@example.com",
    "address": "123 Main St, City, State 12345",
    "website": "www.example.com",
    "company_name": "Example Corp",
    "designation": "CEO"
  },
  "scan_time_ms": 1234
}
```

### GET /
Returns the main web interface.

## Configuration

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key (required)
- `GROQ_MODEL`: Groq model to use (default: meta-llama/llama-4-scout-17b-16e-instruct)

### Image Processing Parameters
- Minimum image dimension: 900px
- Maximum image dimension: 2400px
- Blur detection threshold: 120.0
- Darkness detection threshold: 80.0

### API Settings
- Groq API URL: https://api.groq.com/openai/v1/chat/completions
- Request timeout: 60 seconds
- Temperature: 0.0 (for deterministic outputs)
- Max tokens: 1024

## Troubleshooting

### Common Issues

**API Key Error**
- Ensure GROQ_API_KEY is set in .env file
- Verify your API key is valid and active

**Image Upload Fails**
- Check file format (JPG, PNG supported)
- Ensure file size is reasonable (< 10MB)
- Verify image is not corrupted

**Poor OCR Results**
- Ensure image is clear and well-lit
- Try different image angles
- Check if text is legible
- Consider image resolution (minimum 900px)

**Server Won't Start**
- Check if port 5000 is available
- Verify all dependencies are installed
- Check Python version (3.8+ required)

**Indian Names Not Detected**
- Ensure name is prominent on the card
- Check if font is too stylized or artistic
- Try with higher resolution image
- Verify image is not rotated or skewed

## Performance Optimization

### Image Processing
- Uses fast NLM denoising (searchWindowSize=11) for 4x speed improvement
- Adaptive preprocessing based on image quality
- Efficient variant selection using quality scoring

### API Usage
- Temperature set to 0.0 for consistent, faster responses
- Max tokens limited to 1024 to reduce processing time
- Timeout set to 60 seconds to prevent hanging

### Memory Management
- Images resized to optimal dimensions
- Efficient numpy array operations
- Proper resource cleanup

## Future Enhancements

Potential improvements for the project:
- [ ] Batch processing for multiple cards
- [ ] Export to CSV/Excel
- [ ] Database integration for better storage
- [ ] Support for more languages
- [ ] Advanced error handling and retry logic
- [ ] Mobile app version
- [ ] Integration with contact management systems
- [ ] Support for double-sided cards
- [ ] Real-time processing from camera feed

## License

This project is provided as-is for educational and commercial use.

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues, questions, or suggestions:
- Check the troubleshooting section
- Review the API documentation
- Verify your configuration
- Ensure all dependencies are correctly installed

## Acknowledgments

- **Groq**: For providing the powerful Llama-4 vision model
- **OpenCV**: For the comprehensive computer vision library
- **Flask**: For the lightweight web framework
- **Python Community**: For the extensive ecosystem of libraries

## Technical Details

### Image Preprocessing Pipeline
The application implements a sophisticated 4-stage preprocessing pipeline:
1. **Base Processing**: Denoise → CLAHE → Deskew
2. **Variant Generation**: Creates 4 complementary processed images
3. **Quality Scoring**: Selects the best variant using Laplacian variance and contrast
4. **Zone Detection**: Analyzes spatial relationships and text regions

### Zone Detection Algorithm
1. Convert to grayscale and apply Gaussian blur
2. Apply Otsu thresholding for binary conversion
3. Use morphological operations to identify text regions
4. Classify regions by size and aspect ratio
5. Detect icons and anchors for field identification
6. Extract QR codes using OpenCV's QRCodeDetector

### Prompt Engineering
The system uses carefully crafted prompts to:
- Emphasize name extraction as critical
- Handle Indian name patterns
- Correct common OCR artifacts
- Use spatial relationships for field identification
- Normalize formatting across different card styles

### Error Handling
- Comprehensive exception handling throughout
- Graceful fallback for failed extractions
- Automatic retry with rotation
- JSON validation and normalization
- User-friendly error messages

---

**Built with ❤️ for efficient business card scanning and Indian name detection**
