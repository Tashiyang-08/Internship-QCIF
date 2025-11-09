package com.app;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class EchoController {

  @PostMapping(value = "/echo", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
  public Map<String, Object> echo(@RequestParam("file") MultipartFile file,
                                  @RequestParam(required = false) String language) {
    return Map.of(
        "received", true,
        "filename", file.getOriginalFilename(),
        "size", file.getSize(),
        "language", language == null ? "" : language
    );
  }
}
