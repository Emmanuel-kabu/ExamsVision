-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create exams table
CREATE TABLE IF NOT EXISTS public.exams (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exam_name TEXT NOT NULL,
    exam_type TEXT NOT NULL,
    course_code TEXT NOT NULL,
    department TEXT NOT NULL,
    instructor TEXT NOT NULL,
    degree_type TEXT NOT NULL,
    year_of_study TEXT NOT NULL,
    total_students INTEGER NOT NULL DEFAULT 0,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    duration INTEGER NOT NULL, -- Duration in minutes
    status TEXT NOT NULL DEFAULT 'scheduled', -- scheduled, running, completed, cancelled
    venue TEXT NOT NULL,
    face_detection BOOLEAN NOT NULL DEFAULT true,
    noise_detection BOOLEAN NOT NULL DEFAULT true,
    multi_face_detection BOOLEAN NOT NULL DEFAULT true,
    confidence_threshold FLOAT NOT NULL DEFAULT 0.5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create detections table
CREATE TABLE IF NOT EXISTS public.detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exam_id UUID REFERENCES public.exams(id),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    detection_type TEXT NOT NULL, -- 'cheating' or 'good'
    confidence FLOAT NOT NULL,
    people_count INTEGER NOT NULL,
    camera_id TEXT, -- For multiple camera support
    detection_details JSONB, -- Store additional detection metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create alerts table
CREATE TABLE IF NOT EXISTS public.alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exam_id UUID REFERENCES public.exams(id),
    detection_id UUID REFERENCES public.detections(id),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    alert_type TEXT NOT NULL, -- 'cheating', 'noise', 'multiple_faces', etc.
    confidence FLOAT NOT NULL,
    evidence_path TEXT,
    reviewed BOOLEAN DEFAULT FALSE,
    review_notes TEXT,
    review_timestamp TIMESTAMP WITH TIME ZONE,
    reviewer_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create notifications table
CREATE TABLE IF NOT EXISTS public.notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exam_id UUID REFERENCES public.exams(id),
    alert_id UUID REFERENCES public.alerts(id),
    notification_type TEXT NOT NULL, -- 'alert', 'exam_start', 'exam_end', etc.
    message TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'unread', -- unread, read, archived
    recipient_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    read_at TIMESTAMP WITH TIME ZONE
);

-- Create metrics table for aggregated statistics
CREATE TABLE IF NOT EXISTS public.exam_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exam_id UUID REFERENCES public.exams(id),
    total_detections INTEGER DEFAULT 0,
    cheating_count INTEGER DEFAULT 0,
    good_behavior_count INTEGER DEFAULT 0,
    alert_count INTEGER DEFAULT 0,
    average_confidence FLOAT DEFAULT 0,
    peak_cheating_hour TIME,
    total_monitored_time INTEGER, -- in minutes
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_exams_status ON public.exams(status);
CREATE INDEX IF NOT EXISTS idx_detections_exam_id ON public.detections(exam_id);
CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON public.detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_exam_id ON public.alerts(exam_id);
CREATE INDEX IF NOT EXISTS idx_alerts_reviewed ON public.alerts(reviewed);
CREATE INDEX IF NOT EXISTS idx_notifications_status ON public.notifications(status);
CREATE INDEX IF NOT EXISTS idx_notifications_recipient ON public.notifications(recipient_id);
CREATE INDEX IF NOT EXISTS idx_exam_metrics_exam_id ON public.exam_metrics(exam_id);

-- Function to update exam_metrics
CREATE OR REPLACE FUNCTION update_exam_metrics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update or insert metrics
    INSERT INTO public.exam_metrics (
        exam_id,
        total_detections,
        cheating_count,
        good_behavior_count,
        alert_count,
        average_confidence,
        updated_at
    )
    SELECT 
        NEW.exam_id,
        COUNT(*),
        COUNT(*) FILTER (WHERE detection_type = 'cheating'),
        COUNT(*) FILTER (WHERE detection_type = 'good'),
        (SELECT COUNT(*) FROM public.alerts WHERE exam_id = NEW.exam_id),
        AVG(confidence),
        NOW()
    FROM public.detections
    WHERE exam_id = NEW.exam_id
    ON CONFLICT (exam_id) DO UPDATE
    SET 
        total_detections = EXCLUDED.total_detections,
        cheating_count = EXCLUDED.cheating_count,
        good_behavior_count = EXCLUDED.good_behavior_count,
        alert_count = EXCLUDED.alert_count,
        average_confidence = EXCLUDED.average_confidence,
        updated_at = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update metrics on new detection
CREATE TRIGGER detection_metrics_trigger
AFTER INSERT ON public.detections
FOR EACH ROW
EXECUTE FUNCTION update_exam_metrics();
