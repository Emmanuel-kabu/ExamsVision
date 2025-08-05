# ExamVisio Pro - Refactored Architecture

## 🏗️ Overview

This is a refactored version of the ExamVisio Pro application with improved modular architecture, enhanced error handling, and better maintainability.

## 📁 Project Structure

```
exams/
├── config.py                 # Configuration management
├── video_processor.py        # Video processing and YOLO detection
├── alert_manager.py          # Alert creation and email notifications
├── auth_manager.py           # Authentication and user management
├── data_manager.py           # Data analytics and export functionality
├── app_refactored.py         # Main application (refactored)
├── requirements_refactored.txt # Updated dependencies
└── README_REFACTORED.md      # This file
```

## 🔧 Key Improvements

### 1. **Modular Architecture**
- **Separation of Concerns**: Each component has a specific responsibility
- **Loose Coupling**: Components communicate through well-defined interfaces
- **High Cohesion**: Related functionality is grouped together

### 2. **Enhanced Error Handling**
- **Comprehensive Logging**: Structured logging throughout the application
- **Graceful Degradation**: Application continues working even if some components fail
- **User-Friendly Error Messages**: Clear feedback for users

### 3. **Performance Optimizations**
- **Thread Safety**: Proper threading with locks and events
- **Resource Management**: Automatic cleanup of video resources
- **Memory Management**: Queue-based processing prevents memory leaks

### 4. **Configuration Management**
- **Centralized Config**: All settings in one place with validation
- **Environment Variables**: Secure handling of sensitive data
- **Dynamic Updates**: Real-time parameter changes

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements_refactored.txt
```

### 2. Configure Environment Variables (Optional)

Create a `.env` file:

```env
ALERT_EMAIL=your-email@gmail.com
ALERT_EMAIL_PASSWORD=your-app-password
RECIPIENT_EMAIL=admin@examvisio.com
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
MODEL_PATH=runs/detect/train22/weights/best.pt
```

### 3. Run the Application

```bash
streamlit run app_refactored.py
```

## 📋 Component Details

### Configuration (`config.py`)
- **Centralized Settings**: All configuration in one place
- **Validation**: Parameter validation with helpful error messages
- **Environment Integration**: Automatic loading from environment variables
- **Dynamic Updates**: Real-time configuration changes

```python
from config import config

# Update detection parameters
config.update_detection_params(confidence=0.7, iou=0.5, alert_conf=0.8)

# Get device for model inference
device = config.get_device()  # 'cuda' or 'cpu'
```

### Video Processing (`video_processor.py`)
- **Thread-Safe**: Proper threading with locks and events
- **Multi-Source Support**: Webcam and video file processing
- **Performance Monitoring**: Real-time FPS and processing metrics
- **Resource Cleanup**: Automatic cleanup of video resources

```python
from video_processor import VideoProcessor

processor = VideoProcessor()

# Start monitoring
if processor.start_monitoring(0):  # 0 for webcam
    print("Monitoring started")

# Get performance metrics
metrics = processor.get_performance_metrics()
print(f"Processing time: {metrics['avg_processing_time']:.3f}s")
```

### Alert Management (`alert_manager.py`)
- **Evidence Storage**: Automatic saving of alert images
- **Email Notifications**: HTML email alerts with attachments
- **Alert Filtering**: Filter alerts by confidence, people count, date
- **Statistics**: Comprehensive alert analytics

```python
from alert_manager import AlertManager

alert_mgr = AlertManager()

# Create alert
alert_data = alert_mgr.create_alert(frame, results, people_count)

# Send email alert
alert_mgr.send_email_alert(alert_data)

# Get filtered alerts
filtered = alert_mgr.get_filtered_alerts(min_confidence=0.7, min_people=2)
```

### Authentication (`auth_manager.py`)
- **Multiple Auth Methods**: Local authentication and Google OAuth
- **Session Management**: Secure session handling with expiration
- **Demo Mode**: Fallback when no authentication is configured
- **User Management**: User creation and password management

```python
from auth_manager import AuthManager

auth_mgr = AuthManager()

# Check authentication
if auth_mgr.show_auth_ui():
    user_info = auth_mgr.get_user_info()
    print(f"Welcome, {user_info['name']}")

# Create new user
auth_mgr.create_user("newuser", "New User", "password123")
```

### Data Management (`data_manager.py`)
- **Real-Time Analytics**: Live detection statistics
- **Export Functionality**: CSV, JSON, and Excel export
- **Data Health Monitoring**: Quality metrics and memory usage
- **Backup/Restore**: Data backup and restoration capabilities

```python
from data_manager import DataManager

data_mgr = DataManager()

# Update detection counts
counts = data_mgr.update_counts(results)

# Get analytics
analytics = data_mgr.get_analytics_summary()
print(f"Total detections: {analytics['statistics']['total_detections']}")

# Export data
csv_data = data_mgr.export_history('csv')
```

## 🔍 Features

### Live Monitoring
- **Real-Time Processing**: Live video processing with YOLO detection
- **Multi-Source Support**: Webcam and video file uploads
- **Performance Metrics**: Real-time FPS and processing statistics
- **Alert System**: Automatic cheating detection with evidence capture

### Analytics Dashboard
- **Comprehensive Statistics**: Detection rates, confidence levels, trends
- **Time-Series Analysis**: Historical data visualization
- **Filtering Options**: Time-based and criteria-based filtering
- **Export Capabilities**: Multiple export formats

### Evidence Management
- **Alert Gallery**: Browse all detected incidents
- **Filtering**: Filter by confidence, people count, date
- **Download**: Download evidence images
- **Metadata**: Detailed information about each alert

### System Settings
- **Configuration**: Real-time parameter adjustment
- **Diagnostics**: System health checks
- **Performance Monitoring**: Hardware and software status
- **Model Information**: YOLO model details

## 🛠️ Development

### Adding New Features

1. **Create a new module** in the appropriate component
2. **Update the main app** to integrate the new feature
3. **Add tests** for the new functionality
4. **Update documentation** with usage examples

### Testing

```bash
# Run tests
pytest tests/

# Code formatting
black .

# Linting
flake8 .

# Type checking
mypy .
```

### Deployment

1. **Install dependencies**: `pip install -r requirements_refactored.txt`
2. **Configure environment**: Set up environment variables
3. **Run application**: `streamlit run app_refactored.py`
4. **Monitor logs**: Check application logs for issues

## 🔧 Configuration

### Detection Parameters
- **Confidence Threshold**: Minimum confidence for detections (0.1-0.9)
- **IOU Threshold**: Intersection over Union threshold (0.1-0.9)
- **Alert Confidence**: Minimum confidence for alerts (0.1-0.9)

### Video Processing
- **Frame Size**: Resolution for processing (320x240 to 1920x1080)
- **Target FPS**: Desired frame rate (1-30)
- **Queue Size**: Frame buffer size (5-20)

### Alert Settings
- **Email Alerts**: Enable/disable email notifications
- **Alert Cooldown**: Time between alerts (1-60 seconds)
- **SMTP Settings**: Email server configuration

## 🐛 Troubleshooting

### Common Issues

1. **Camera Not Working**
   - Check camera permissions
   - Try different camera index (0, 1, 2)
   - Verify camera is not in use by another application

2. **Model Loading Failed**
   - Verify model file exists at specified path
   - Check CUDA availability for GPU inference
   - Ensure sufficient memory for model loading

3. **Email Alerts Not Working**
   - Verify SMTP settings
   - Check email credentials
   - Ensure network connectivity

4. **Performance Issues**
   - Reduce frame size
   - Lower target FPS
   - Use CPU instead of GPU if memory is limited

### Logging

The application uses structured logging. Check logs for detailed error information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster inference
2. **Optimize Frame Size**: Smaller frames process faster
3. **Adjust FPS**: Lower FPS reduces CPU usage
4. **Monitor Memory**: Keep an eye on memory usage during long sessions
5. **Regular Cleanup**: Clear old data periodically

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests**
5. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review the logs for error details
- Create an issue with detailed information
- Contact the development team

---

**Note**: This refactored version maintains backward compatibility while providing enhanced functionality, better error handling, and improved maintainability. 