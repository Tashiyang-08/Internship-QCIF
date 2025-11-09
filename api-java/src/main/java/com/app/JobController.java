package com.app;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.ContentDisposition;
import org.springframework.http.HttpEntity;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class JobController {

  // FastAPI base URL
  private final RestClient stt = RestClient.create("http://localhost:8001");

  @PostMapping(value = "/jobs", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
  public @ResponseBody Object createJob(
      @RequestPart("file") MultipartFile file,
      @RequestParam(required = false) String language,
      @RequestParam(required = false, name = "model") String modelName,
      @RequestParam(name = "pretty_time", defaultValue = "false") boolean prettyTime,
      @RequestParam(name = "round_secs", defaultValue = "2") int roundSecs
  ) {
    try {
      // try common part names FastAPI apps use
      List<String> partNames = List.of("file", "audio", "audio_file");

      Map resp = null;
      Integer lastStatus = null;
      String lastBody = null;

      for (String partName : partNames) {
        try {
          resp = forwardToFastApi(file, language, modelName, prettyTime, roundSecs, partName);
          break; // success
        } catch (HttpClientErrorException e) {
          lastStatus = e.getStatusCode().value();
          lastBody = e.getResponseBodyAsString();
          if (e.getStatusCode().value() != 422) {
            throw e; // not a validation issue
          }
          // otherwise retry with next partName
        }
      }

      if (resp == null) {
        return Map.of(
            "error", "bad_request_to_stt",
            "message", "FastAPI rejected the multipart. Tried part names: " + partNames,
            "lastStatus", lastStatus,
            "lastBody", lastBody
        );
      }

      return resp;

    } catch (HttpClientErrorException e) {
      return Map.of(
          "error", "stt_http_error",
          "status", e.getStatusCode().value(),
          "body", e.getResponseBodyAsString()
      );
    } catch (Exception e) {
      return Map.of(
          "error", "stt_call_failed",
          "message", e.getMessage()
      );
    }
  }

  private Map forwardToFastApi(
      MultipartFile file,
      String language,
      String modelName,
      boolean prettyTime,
      int roundSecs,
      String partName
  ) throws Exception {

    // ---- filename must be final for the inner class
    String filenameTmp = file.getOriginalFilename();
    if (filenameTmp == null || filenameTmp.isBlank()) filenameTmp = "upload.bin";
    final String filename = filenameTmp;

    // ---- file content
    byte[] bytes = file.getBytes();

    // ---- explicit content type (fallback to octet-stream)
    MediaType ct;
    try {
      ct = (file.getContentType() != null)
          ? MediaType.parseMediaType(file.getContentType())
          : MediaType.APPLICATION_OCTET_STREAM;
    } catch (Exception ignore) {
      ct = MediaType.APPLICATION_OCTET_STREAM;
    }

    // ---- build the file part with filename
    ByteArrayResource resource = new ByteArrayResource(bytes) {
      @Override public String getFilename() { return filename; }
    };

    var fileHeaders = new org.springframework.http.HttpHeaders();
    fileHeaders.setContentType(ct);
    fileHeaders.setContentDisposition(ContentDisposition.builder("form-data")
        .name(partName)
        .filename(filename)
        .build());
    HttpEntity<ByteArrayResource> filePart = new HttpEntity<>(resource, fileHeaders);

    // ---- optional text parts as text/plain
    var textHeaders = new org.springframework.http.HttpHeaders();
    textHeaders.setContentType(MediaType.TEXT_PLAIN);

    MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
    body.add(partName, filePart);
    if (language != null && !language.isBlank()) {
      body.add("language", new HttpEntity<>(language, textHeaders));
    }
    if (modelName != null && !modelName.isBlank()) {
      body.add("model", new HttpEntity<>(modelName, textHeaders));
    }

    // ---- call FastAPI with the query params that worked in your logs
    return stt.post()
        .uri(uriBuilder -> uriBuilder
            .path("/transcribe")
            .queryParam("pretty_time", prettyTime)
            .queryParam("round_secs", roundSecs)
            .build())
        .contentType(MediaType.MULTIPART_FORM_DATA)
        .accept(MediaType.APPLICATION_JSON)
        .body(body)
        .retrieve()
        .body(Map.class);
  }
}
