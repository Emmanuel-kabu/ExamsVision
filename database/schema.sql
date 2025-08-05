-- Exams table to store exam information
CREATE TABLE exams (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    exam_name VARCHAR(255) NOT NULL,
    course_code VARCHAR(50) NOT NULL,
    department VARCHAR(100) NOT NULL,
    instructor VARCHAR(255) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    duration INTEGER NOT NULL, -- in minutes
    status VARCHAR(20) CHECK (status IN ('scheduled', 'running', 'completed', 'cancelled')) DEFAULT 'scheduled',
    total_students INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Detections table to store real-time detection data
CREATE TABLE detections (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    exam_id UUID REFERENCES exams(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    detection_type VARCHAR(20) CHECK (detection_type IN ('cheating', 'good')),
    confidence DECIMAL(4,3) NOT NULL,
    people_count INTEGER NOT NULL,
    frame_id VARCHAR(100), -- Reference to stored frame if any
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Alerts table to store generated alerts
CREATE TABLE alerts (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    exam_id UUID REFERENCES exams(id),
    detection_id UUID REFERENCES detections(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    alert_type VARCHAR(50) NOT NULL,
    confidence DECIMAL(4,3) NOT NULL,
    evidence_path VARCHAR(500), -- Path to stored evidence image
    reviewed BOOLEAN DEFAULT FALSE,
    review_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Users table for authentication and authorization
CREATE TABLE users (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    encrypted_password VARCHAR(255) NOT NULL,
    role VARCHAR(20) CHECK (role IN ('admin', 'instructor', 'proctor')) NOT NULL,
    department VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ExamProctors table to associate proctors with exams
CREATE TABLE exam_proctors (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    exam_id UUID REFERENCES exams(id),
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(exam_id, user_id)
);

-- Create indexes for better query performance
CREATE INDEX idx_exams_status ON exams(status);
CREATE INDEX idx_detections_exam_id ON detections(exam_id);
CREATE INDEX idx_detections_timestamp ON detections(timestamp);
CREATE INDEX idx_alerts_exam_id ON alerts(exam_id);
CREATE INDEX idx_alerts_timestamp ON alerts(timestamp);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updating timestamps
CREATE TRIGGER update_exams_updated_at
    BEFORE UPDATE ON exams
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
