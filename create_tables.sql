CREATE TABLE public.exams (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    exam_name TEXT NOT NULL,
    exam_type TEXT NOT NULL,
    course_code TEXT NOT NULL,
    department TEXT NOT NULL,
    instructor TEXT NOT NULL,
    degree_type TEXT NOT NULL,
    year_of_study TEXT NOT NULL,
    total_students INTEGER NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    duration INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'scheduled',
    venue TEXT NOT NULL,
    face_detection BOOLEAN NOT NULL DEFAULT true,
    noise_detection BOOLEAN NOT NULL DEFAULT true,
    multi_face_detection BOOLEAN NOT NULL DEFAULT true,
    confidence_threshold FLOAT NOT NULL DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT now()
     );

-- Create the scheduled_exams table
CREATE TABLE public.scheduled_exams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exam_id UUID REFERENCES public.exams(id) ON DELETE CASCADE,
    exam_name TEXT NOT NULL,
    course_code TEXT NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    venue TEXT NOT NULL,
    instructor TEXT NOT NULL,
    total_students INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'scheduled',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Create the detections table
CREATE TABLE public.detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exam_id UUID REFERENCES public.exams(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
    detection_type TEXT NOT NULL,
    confidence FLOAT,
    people_count INTEGER,
    details JSONB
);

-- Create the alerts table
CREATE TABLE public.alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exam_id UUID REFERENCES public.exams(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(),
    alert_type TEXT NOT NULL,
    confidence FLOAT,
    evidence_path TEXT,
    details JSONB,
    reviewed BOOLEAN DEFAULT false
);

-- Create the exam_metrics table for storing detailed statistics
CREATE TABLE public.exam_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exam_id UUID REFERENCES public.exams(id) ON DELETE CASCADE,
    total_detections INTEGER NOT NULL DEFAULT 0,
    total_cheating INTEGER NOT NULL DEFAULT 0,
    total_good_behavior INTEGER NOT NULL DEFAULT 0,
    cheating_percentage FLOAT,
    good_behavior_percentage FLOAT,
    mean_confidence FLOAT,
    median_confidence FLOAT,
    max_confidence FLOAT,
    min_confidence FLOAT,
    std_deviation FLOAT,
    peak_cheating_hour TIME,
    lowest_cheating_hour TIME,
    alert_frequency FLOAT,
    average_response_time FLOAT,
    total_alerts INTEGER,
    resolved_alerts INTEGER,
    alert_resolution_rate FLOAT,
    attendance_rate FLOAT,
    average_students_present INTEGER,
    monitoring_duration INTERVAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT now(),
    metrics_data JSONB -- For storing additional metrics
);

-- Create the exam_reports table for storing generated reports
CREATE TABLE public.exam_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exam_id UUID REFERENCES public.exams(id) ON DELETE CASCADE,
    report_type TEXT NOT NULL, -- 'summary', 'detailed', 'incident', etc.
    report_format TEXT NOT NULL, -- 'pdf', 'csv', 'json', etc.
    report_data JSONB NOT NULL, -- Store the actual report data
    file_path TEXT, -- Path to stored report file if applicable
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    generated_by TEXT, -- User who generated the report
    report_version TEXT,
    is_final BOOLEAN DEFAULT false,
    metadata JSONB -- Additional report metadata
);

-- Create indexes for better performance
CREATE INDEX idx_scheduled_exams_start_time ON public.scheduled_exams(start_time);
CREATE INDEX idx_scheduled_exams_end_time ON public.scheduled_exams(end_time);
CREATE INDEX idx_scheduled_exams_status ON public.scheduled_exams(status);
CREATE INDEX idx_detections_exam_id ON public.detections(exam_id);
CREATE INDEX idx_detections_timestamp ON public.detections(timestamp);
CREATE INDEX idx_alerts_exam_id ON public.alerts(exam_id);
CREATE INDEX idx_alerts_timestamp ON public.alerts(timestamp);
CREATE INDEX idx_alerts_reviewed ON public.alerts(reviewed);
CREATE INDEX idx_exam_metrics_exam_id ON public.exam_metrics(exam_id);
CREATE INDEX idx_exam_reports_exam_id ON public.exam_reports(exam_id);
CREATE INDEX idx_exam_reports_generated_at ON public.exam_reports(generated_at);

-- Add triggers to update last_updated timestamp
CREATE OR REPLACE FUNCTION update_last_updated_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_exams_last_updated
    BEFORE UPDATE ON public.exams
    FOR EACH ROW
    EXECUTE FUNCTION update_last_updated_column();

CREATE TRIGGER update_scheduled_exams_last_updated
    BEFORE UPDATE ON public.scheduled_exams
    FOR EACH ROW
    EXECUTE FUNCTION update_last_updated_column();

CREATE TRIGGER update_exam_metrics_last_updated
    BEFORE UPDATE ON public.exam_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_last_updated_column();
