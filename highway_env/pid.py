import time

class PID:
    
    def __init__(self,
                 K_P: float,
                 K_I: float,
                 K_D: float) -> None:
        
        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D
        self.prev_error = 0
        self.integral_error = 0
        self.last_time = time.perf_counter()
        
    def clear(self):
        self.prev_error = 0
        self.integral_error = 0
        
    
    def get_value(self, value, target_value, error = None):
        
        if error is None:
            error = target_value - value
            
        t_m = time.perf_counter()
        d_error = (error - self.prev_error)/(t_m-self.last_time)
        i_error= self.integral_error + error*(t_m-self.last_time)
        t = self.K_P * error + self.K_D * d_error + self.K_I * i_error
        self.prev_error = error 
        self.integral_error = i_error
        self.last_time = t_m
                
        return t