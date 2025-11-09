package com.app;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.*;

@Configuration
public class CorsConfig implements WebMvcConfigurer {
  @Override
  public void addCorsMappings(CorsRegistry r) {
    r.addMapping("/api/**")
     .allowedOrigins("http://localhost:5173", "http://127.0.0.1:5173")
     .allowedMethods("GET","POST","PUT","PATCH","DELETE","OPTIONS")
     .allowedHeaders("*")
     .allowCredentials(true)
     .maxAge(3600);
  }
}
