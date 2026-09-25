#' Discover and Select Torch Execution Devices
#'
#' `riem.devices()` reports the devices visible to the current R process and
#' checks whether they support riemtorch's core linear-algebra operations at the
#' requested precision. `riem.device()` applies the package selection policy
#' and returns one reusable torch device.
#'
#' @param dtype Optional floating-point torch dtype, or one of `"float64"`,
#'   `"float32"`, `"float16"`, `"bfloat16"`, `"complex64"`, or
#'   `"complex128"`. Discovery defaults to
#'   float64.
#' @param device A device request. `NULL` uses, in order,
#'   `getOption("riemtorch.device")`, `RIEMTORCH_DEVICE`, and `"auto"`.
#'   Explicit `"auto"` ignores those defaults. Use `"cpu"`, `"mps"`,
#'   `"cuda"`, or a zero-based CUDA index such as `"cuda:1"` to override.
#' @param operations Required device operations: a subset of `basic`, `cholesky`,
#'   `eigh`, `svd`, `qr`, `solve`, and `matrix_exp`. NULL probes all.
#' @param reference Optional tensor or tensor tree whose floating-point dtype is
#'   preserved when `dtype` is omitted.
#' @return `riem.devices()` returns a data frame. `riem.device()` returns a
#'   `torch_device`.
#' @details Automatic selection tries the current visible CUDA device, the
#'   remaining visible CUDA devices in index order, compatible MPS, and CPU.
#'   A small computation probe checks tensor creation, autograd, decompositions,
#'   and the matrix exponential. An automatic request skips incompatible
#'   accelerators; an explicit unavailable or incompatible request is an error.
#'   No package-load device initialization or runtime download is performed.
#' @examples
#' if (torch::torch_is_installed()) {
#'   riem.devices()
#'   riem.device("cpu")
#' }
#' @export
riem.devices <- function(dtype = NULL, operations = NULL) {
  dtype <- .normalize_dtype(dtype, NULL)
  candidates <- .visible_devices()
  rows <- lapply(candidates, function(device) {
    probe <- .probe_execution(device, dtype, operations)
    data.frame(
      device = .device_string(device),
      type = device$type,
      index = if (is.null(device$index)) NA_integer_ else as.integer(device$index),
      available = TRUE,
      compatible = probe$ok,
      reason = probe$reason,
      model = .device_model(device),
      operations = paste(.probe_operations(operations),collapse=","),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  attr(out, "dtype") <- .dtype_string(dtype)
  out
}

#' @rdname riem.devices
#' @export
riem.device <- function(device = NULL, dtype = NULL, reference = NULL, operations = NULL) {
  .resolve_execution(device = device, dtype = dtype, reference = reference, operations=operations)$device
}

#' Move Tensor Trees and Device-Bound Geometry
#'
#' Move a tensor tree, registered problem data, or built-in geometry to one
#' resolved device. Floating-point tensors use the resolved dtype; integer and
#' logical tensors retain their dtype.
#'
#' @param x A torch tensor, nested list, `riem_manifold`, or `riem_problem`.
#' @param device,dtype Device and floating-point dtype requests as described in
#'   [riem.devices()].
#' @return An object with the same structure and moved tensor values.
#' @details Generalized Stiefel and Grassmann geometries are rebuilt so that
#'   their fixed SPD matrix and derived whitening operators share the target
#'   placement. Product factors are converted recursively. Other built-in
#'   geometries contain no device-bound state. Registered problem data and
#'   least-squares weights are moved, but tensors captured privately inside a
#'   callback remain caller-owned and must be registered as `data` instead.
#'   Custom manifolds with tensors in their specification cannot be rebuilt
#'   safely and produce an explicit error.
#' @examples
#' if (torch::torch_is_installed()) {
#'   values <- list(x = torch::torch_ones(2, dtype = torch::torch_float64()),
#'                  index = torch::torch_tensor(1:2, dtype = torch::torch_int64()))
#'   moved <- riem.to(values, device = "cpu", dtype = "float32")
#'   moved$x$dtype
#'   moved$index$dtype
#' }
#' @export
riem.to <- function(x, device = NULL, dtype = NULL) {
  reference <- .placement_reference(x)
  execution <- .resolve_execution(device = device, dtype = dtype,
                                  reference = reference)
  .to_execution(x, execution)
}

.floating_dtypes <- function() list(
  float64 = torch::torch_float64(),
  float32 = torch::torch_float32(),
  float16 = torch::torch_float16(),
  bfloat16 = torch::torch_bfloat16(),
  complex64 = torch::torch_cfloat(), complex128 = torch::torch_cdouble()
)

.normalize_dtype <- function(dtype, reference = NULL) {
  if (is.null(dtype)) {
    if (inherits(reference, "torch_tensor") && (reference$is_floating_point()||reference$is_complex()))
      return(reference$dtype)
    return(torch::torch_float64())
  }
  choices <- .floating_dtypes()
  if (is.character(dtype)) {
    if (length(dtype) != 1L || is.na(dtype) || !nzchar(dtype))
      .stop("dtype must be one floating-point dtype")
    key <- tolower(gsub("^torch[._]", "", dtype))
    aliases <- c(double = "float64", float = "float32", half = "float16")
    if (key %in% names(aliases)) key <- unname(aliases[[key]])
    if (!key %in% names(choices))
      .stop("Unsupported dtype request: ", dtype)
    return(choices[[key]])
  }
  if (!torch::is_torch_dtype(dtype))
    .stop("dtype must be a floating-point torch dtype or supported name")
  if (!any(vapply(choices, function(candidate) isTRUE(dtype == candidate),
                  logical(1))))
    .stop("dtype must be floating point")
  dtype
}

.dtype_string <- function(dtype) {
  choices <- .floating_dtypes()
  hit <- names(choices)[vapply(choices, function(candidate)
    isTRUE(dtype == candidate), logical(1))]
  if (!length(hit)) tolower(as.character(dtype)) else hit[[1L]]
}

.device_string <- function(device) {
  if (identical(device$type, "cuda") && !is.null(device$index))
    return(paste0("cuda:", as.integer(device$index)))
  device$type
}

.normalize_device <- function(device) {
  if (torch::is_torch_device(device)) {
    type <- device$type
    index <- device$index
  } else {
    if (!is.character(device) || length(device) != 1L || is.na(device) ||
        !nzchar(device)) .stop("device must be one device name")
    request <- tolower(trimws(device))
    if (request == "auto") return("auto")
    if (!grepl("^(cpu|mps|cuda)(:[0-9]+)?$", request))
      .stop("Unsupported device request: ", device)
    pieces <- strsplit(request, ":", fixed = TRUE)[[1L]]
    type <- pieces[[1L]]
    index <- if (length(pieces) == 2L) as.integer(pieces[[2L]]) else NULL
  }
  if (!type %in% c("cpu", "cuda", "mps"))
    .stop("Unsupported device type: ", type)
  if (!identical(type, "cuda") && !is.null(index))
    .stop("Only CUDA devices accept an index")
  if (identical(type, "cuda") && is.null(index) &&
      isTRUE(tryCatch(torch::cuda_is_available(), error = function(e) FALSE)))
    index <- tryCatch(as.integer(torch::cuda_current_device()),
                      error = function(e) NULL)
  torch::torch_device(type, index = index)
}

.device_request <- function(device) {
  if (!is.null(device))
    return(list(value = device, source = "argument"))
  option <- getOption("riemtorch.device", NULL)
  if (!is.null(option))
    return(list(value = option, source = "option"))
  environment <- Sys.getenv("RIEMTORCH_DEVICE", unset = "")
  if (nzchar(environment))
    return(list(value = environment, source = "environment"))
  list(value = "auto", source = "default")
}

.cuda_devices <- function() {
  available <- isTRUE(tryCatch(torch::cuda_is_available(),
                               error = function(e) FALSE))
  if (!available) return(list())
  count <- tryCatch(as.integer(torch::cuda_device_count()),
                    error = function(e) 0L)
  if (!is.finite(count) || count < 1L) return(list())
  indices <- seq.int(0L, count - 1L)
  current <- tryCatch(as.integer(torch::cuda_current_device()),
                      error = function(e) indices[[1L]])
  if (current %in% indices) indices <- c(current, indices[indices != current])
  lapply(indices, function(index) torch::torch_device("cuda", index = index))
}

.mps_available <- function() {
  isTRUE(tryCatch(torch::backends_mps_is_available(),
                  error = function(e) FALSE))
}

.visible_devices <- function() {
  candidates <- .cuda_devices()
  if (.mps_available()) candidates <- c(candidates, list(torch::torch_device("mps")))
  c(candidates, list(torch::torch_device("cpu")))
}

.device_probe_cache <- new.env(parent = emptyenv())

.device_available <- function(device) {
  if (identical(device$type, "cpu")) return(TRUE)
  if (identical(device$type, "mps")) return(.mps_available())
  if (!identical(device$type, "cuda")) return(FALSE)
  if (!isTRUE(tryCatch(torch::cuda_is_available(), error = function(e) FALSE)))
    return(FALSE)
  count <- tryCatch(as.integer(torch::cuda_device_count()),
                    error = function(e) 0L)
  index <- device$index
  if (is.null(index)) return(count > 0L)
  is.finite(index) && index >= 0L && index < count
}

.probe_execution <- function(device, dtype, operations=NULL) {
  operations <- .probe_operations(operations)
  if (!.device_available(device))
    return(list(ok = FALSE, reason = "device is not available to this R process"))
  key <- paste(.device_string(device), .dtype_string(dtype), paste(operations,collapse=","), sep = "/")
  if (exists(key, envir = .device_probe_cache, inherits = FALSE))
    return(get(key, envir = .device_probe_cache, inherits = FALSE))
  result <- tryCatch({
    torch::with_enable_grad({
      x <- torch::torch_tensor(matrix(c(2, 0, 0, 3), 2, 2),
        dtype = dtype, device = device, requires_grad = TRUE)
      identity <- torch::torch_eye(2, dtype = dtype, device = device)
      positive <- .adj(x)$matmul(x) + identity
      if("cholesky" %in% operations) torch::linalg_cholesky(positive)
      if("eigh" %in% operations) torch::linalg_eigh(positive)
      if("svd" %in% operations) torch::linalg_svd(x)
      if("qr" %in% operations) torch::linalg_qr(x)
      if("solve" %in% operations) torch::linalg_solve(positive,identity)
      value <- if("matrix_exp" %in% operations) torch::torch_matrix_exp(x / 10)$sum() else (x*x)$sum()
      if(value$is_complex()) value<-value$real
      gradient <- torch::autograd_grad(value, x)[[1L]]
      if (!isTRUE(as.logical(torch::torch_isfinite(gradient)$all()$item())))
        stop("probe returned a nonfinite gradient")
    })
    list(ok = TRUE, reason = paste("passed operations:",paste(operations,collapse=", ")))
  }, error = function(e) list(ok = FALSE,
    reason = gsub("[\r\n]+", " ", conditionMessage(e))))
  assign(key, result, envir = .device_probe_cache)
  result
}

.resolve_execution <- function(device = NULL, dtype = NULL, reference = NULL, operations=NULL) {
  reference <- .placement_reference(reference)
  dtype <- .normalize_dtype(dtype, reference)
  request <- .device_request(device)
  normalized <- .normalize_device(request$value)
  requested <- if (identical(normalized, "auto")) "auto" else
    .device_string(normalized)
  if (identical(normalized, "auto")) {
    failures <- character()
    for (candidate in .visible_devices()) {
      probe <- .probe_execution(candidate, dtype, operations)
      if (probe$ok) {
        selected <- candidate
        reason <- paste0("automatic selection: ", probe$reason)
        break
      }
      failures <- c(failures, paste0(.device_string(candidate), ": ",
                                     probe$reason))
    }
    if (!exists("selected", inherits = FALSE))
      .stop("No compatible torch device was found for ", .dtype_string(dtype),
            ". ", paste(failures, collapse = "; "))
  } else {
    selected <- normalized
    if (!.device_available(selected))
      .stop("Requested device ", .device_string(selected),
            " is not available to this R process")
    probe <- .probe_execution(selected, dtype, operations)
    if (!probe$ok)
      .stop("Requested device ", .device_string(selected), " is incompatible with ",
            .dtype_string(dtype), ": ", probe$reason)
    reason <- paste0("explicit selection: ", probe$reason)
  }
  list(requested = requested, source = request$source, device = selected,
       dtype = dtype, device_string = .device_string(selected),
       dtype_string = .dtype_string(dtype), reason = reason,
       model=.device_model(selected), operations=.probe_operations(operations))
}

.placement_reference <- function(x) {
  if (inherits(x, "torch_tensor")) {
    if ((x$is_floating_point()||x$is_complex())) return(x)
    return(NULL)
  }
  if (inherits(x, "riem_problem")) {
    for (candidate in list(x$data, x$weights, x$manifold)) {
      found <- .placement_reference(candidate)
      if (!is.null(found)) return(found)
    }
    return(NULL)
  }
  if (inherits(x, "riem_product")) {
    for (factor in x$factors) {
      found <- .placement_reference(factor)
      if (!is.null(found)) return(found)
    }
    return(NULL)
  }
  if (inherits(x, "riem_manifold")) {
    if (x$name %in% c("stiefel.generalized", "grassmann.generalized"))
      return(x$specification$B)
    return(NULL)
  }
  if (is.list(x)) {
    for (element in x) {
      found <- .placement_reference(element)
      if (!is.null(found)) return(found)
    }
  }
  NULL
}

.contains_tensor <- function(x) {
  if (inherits(x, "torch_tensor")) return(TRUE)
  if (!is.list(x)) return(FALSE)
  any(vapply(x, .contains_tensor, logical(1)))
}

.to_execution <- function(x, execution) {
  if (inherits(x, "torch_tensor")) {
    if ((x$is_floating_point()||x$is_complex()))
      return(x$to(device = execution$device, dtype = execution$dtype))
    return(x$to(device = execution$device))
  }
  if (inherits(x, "riem_problem")) {
    out <- x
    out$manifold <- .to_execution(x$manifold, execution)
    if (!is.null(x$data)) out$data <- .to_execution(x$data, execution)
    if (!is.null(x$weights)) out$weights <- .to_execution(x$weights, execution)
    return(out)
  }
  if (inherits(x, "riem_product")) {
    out <- x
    out$factors <- lapply(x$factors, .to_execution, execution = execution)
    return(out)
  }
  if (inherits(x, "riem_manifold")) {
    if(x$name=="power") return(manifold.power(.to_execution(x$base,execution),x$copies))
    if(x$name=="scaled") return(manifold.scaled(.to_execution(x$base,execution),x$scale))
    if(x$name=="affine") return(manifold.affine(.to_execution(x$specification$basis,execution),.to_execution(x$specification$offset,execution)))
    if (x$name %in% c("stiefel.generalized", "grassmann.generalized")) {
      B <- .to_execution(x$specification$B, execution)
      if (x$name == "stiefel.generalized")
        return(manifold.stiefel.generalized(x$specification$p,
          x$specification$k, B))
      return(manifold.grassmann.generalized(x$specification$p,
        x$specification$k, B))
    }
    if (.contains_tensor(x$specification))
      .stop("Custom device-bound manifold state cannot be migrated safely; rebuild the manifold on the target device")
    return(x)
  }
  if (is.list(x)) {
    out <- lapply(x, .to_execution, execution = execution)
    attributes(out) <- attributes(x)
    return(out)
  }
  x
}

.probe_operations <- function(operations=NULL) {
  all <- c("basic","cholesky","eigh","svd","qr","solve","matrix_exp")
  if(is.null(operations)) return(all)
  if(!is.character(operations)||anyNA(operations)||any(!operations %in% all))
    .stop("Unknown device probe operations")
  sort(unique(c("basic",operations)))
}
.required_operations <- function(M) {
  if(inherits(M,"riem_product")) return(unique(unlist(lapply(M$factors,.required_operations))))
  .probe_operations(M$capabilities$required_operations)
}
.model_cache<-new.env(parent=emptyenv())
.device_model <- function(device) {
  key<-.device_string(device)
  if(exists(key,.model_cache,inherits=FALSE)) return(get(key,.model_cache))
  model<-paste(Sys.info()[["machine"]],"CPU")
  if(device$type=="cuda") {
    model<-"CUDA (model unavailable through R torch)"
    executable<-Sys.which("nvidia-smi")
    if(nzchar(executable)) {
      inventory<-tryCatch(suppressWarnings(system2(executable,
        c("--query-gpu=index,uuid,name","--format=csv,noheader"),stdout=TRUE,stderr=FALSE)),error=function(e) character())
      if(length(inventory)&&is.null(attr(inventory,"status")))
        model<-paste("Host GPU inventory (visible-to-physical index mapping not inferred):",paste(inventory,collapse="; "))
    }
  } else if(device$type=="mps") model<-"Apple Metal" else {
    if(Sys.info()[["sysname"]]=="Darwin"&&nzchar(Sys.which("sysctl"))) {
      brand<-tryCatch(suppressWarnings(system2(Sys.which("sysctl"),c("-n","machdep.cpu.brand_string"),stdout=TRUE,stderr=FALSE)),error=function(e) character())
      if(length(brand)&&is.null(attr(brand,"status"))) model<-brand[1]
    } else if(file.exists("/proc/cpuinfo")) {
      lines<-readLines("/proc/cpuinfo",n=100,warn=FALSE);brand<-lines[grepl("^model name",lines)]
      if(length(brand)) model<-sub("^[^:]*: *","",brand[1])
    } else if(nzchar(Sys.getenv("PROCESSOR_IDENTIFIER"))) model<-Sys.getenv("PROCESSOR_IDENTIFIER")
  }
  if(device$type=="cpu"&&!grepl("CPU",model,fixed=TRUE)) model<-paste(model,"CPU")
  assign(key,model,.model_cache);model
}
.execution_metadata <- function(execution) list(
  requested=execution$requested,source=execution$source,device=execution$device_string,
  dtype=execution$dtype_string,reason=execution$reason,model=execution$model,
  operations=execution$operations,R=as.character(getRversion()),
  torch=as.character(utils::packageVersion("torch")),
  libtorch=.torch_runtime_version(),
  cuda_runtime=if(execution$device$type=="cuda") tryCatch(as.character(torch::cuda_runtime_version()),error=function(e) NA_character_) else NA_character_,
  cuda_visible_devices=if(execution$device$type=="cuda") Sys.getenv("CUDA_VISIBLE_DEVICES",unset=NA_character_) else NA_character_)

.torch_runtime_version <- function() {
  path<-system.file("share","cmake","Torch","TorchConfigVersion.cmake",package="torch")
  if(!nzchar(path)) return(NA_character_)
  lines<-readLines(path,warn=FALSE)
  hit<-lines[grepl("set\\(PACKAGE_VERSION ",lines)]
  if(!length(hit)) return(NA_character_)
  sub('.*"([^" ]+)".*','\\1',hit[1])
}
