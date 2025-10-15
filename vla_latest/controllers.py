"""
控制器模块
包含PID控制器等基础控制算法
"""


class PIDController:
    """PID控制器"""
    
    def __init__(self, kp=0.1, ki=0.0, kd=0.0, max_output=1.0, min_output=-1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.min_output = min_output
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.033
        
    def update(self, error):
        """更新PID控制器"""
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0.0
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(self.min_output, min(self.max_output, output))
        
        self.prev_error = error
        return output
    
    def reset(self):
        """重置PID控制器状态"""
        self.integral = 0.0
        self.prev_error = 0.0