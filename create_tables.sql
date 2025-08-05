-- Create exams table if not exists
CREATE TABLE IF NOT EXISTS exams (
    id UUID PRIMARY KEY,
    exam_name TEXT NOT NULL,
    course_code TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    status TEXT NOT NULL,
    total_students INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create detections table if not exists
CREATE TABLE IF NOT EXISTS detections (
    id UUID PRIMARY KEY,
    exam_id UUID REFERENCES exams(id),
    timestamp TIMESTAMP NOT NULL,
    detection_type TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    image_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create alerts table if not exists
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY,
    exam_id UUID REFERENCES exams(id),
    timestamp TIMESTAMP NOT NULL,
    alert_type TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    evidence_path TEXT NOT NULL,
    reviewed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
