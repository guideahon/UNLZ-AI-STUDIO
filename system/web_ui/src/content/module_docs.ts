export type ModuleDoc = {
  title: string;
  summary: string;
  what_is: string;
  purpose: string;
  use_cases: string[];
  how_to: string[];
};

export const MODULE_DOCS: Record<string, ModuleDoc> = {
  monitor: {
    title: "Endpoints de IA",
    summary:
      "Panel operativo para revisar hardware y controlar servicios locales de IA con estado, puertos y telemetria.",
    what_is:
      "Dashboard local que detecta CPU/GPU/RAM y orquesta servicios (LLM/CLM/VLM/ALM/SLM) desde una sola vista.",
    purpose:
      "Centralizar el control para evitar usar terminales, validar que el entorno esta listo y diagnosticar cuellos de botella.",
    use_cases: [
      "Verificar la GPU activa y su memoria antes de iniciar un trabajo pesado.",
      "Encender o detener un servidor LLM sin salir del estudio.",
      "Confirmar que el puerto del servicio esta disponible.",
      "Controlar temperatura y uso en tareas de entrenamiento o inferencia.",
      "Liberar recursos deteniendo servicios que no se usan.",
    ],
    how_to: [
      "Abrir Endpoints de IA desde el menu principal.",
      "Revisar las tarjetas de CPU/GPU/RAM para confirmar recursos.",
      "Elegir el servicio a iniciar (LLM/CLM/VLM/ALM/SLM).",
      "Seleccionar modelo y parametros basicos cuando aplique.",
      "Presionar Iniciar y esperar el estado activo.",
      "Verificar puerto y estado en la lista.",
      "Detener el servicio cuando finalice el uso.",
    ],
  },
  llm_frontend: {
    title: "Chat AI",
    summary:
      "Chat local con modelos GGUF, historial y gestor de modelos para trabajar sin depender de internet.",
    what_is:
      "Interfaz web para conversar con modelos locales y administrar descargas, perfiles y parametros.",
    purpose:
      "Ofrecer asistencia offline, comparar modelos y ajustar configuraciones sin salir del entorno.",
    use_cases: [
      "Asistente de programacion local y privado.",
      "Redaccion y resumen de textos sin servicios externos.",
      "Comparar respuestas entre modelos GGUF.",
      "Probar prompts y plantillas institucionales.",
      "Generar borradores para clases o informes.",
    ],
    how_to: [
      "Ir a Chat AI desde el menu.",
      "Seleccionar un modelo disponible.",
      "Si el servidor no esta activo, iniciarlo desde Endpoints.",
      "Ajustar temperatura, top-p o max tokens si aplica.",
      "Escribir el prompt y enviar.",
      "Guardar o reutilizar conversaciones del historial.",
      "Usar la seccion de descargas para agregar nuevos GGUF.",
    ],
  },
  inclu_ia: {
    title: "Inclu-IA",
    summary:
      "Subtitulado en tiempo real para clases accesibles, con servidor local y salida web.",
    what_is:
      "Servicio STT local con configuracion de modelo, idioma y salida en vivo.",
    purpose:
      "Mejorar accesibilidad y registrar clases o reuniones con transcripcion automatica.",
    use_cases: [
      "Subtitulado en clases presenciales.",
      "Transcripcion en laboratorios o talleres.",
      "Accesibilidad para estudiantes con hipoacusia.",
      "Registro de charlas internas o seminarios.",
    ],
    how_to: [
      "Elegir el tamano del modelo (Air/Nano si aplica).",
      "Seleccionar idioma y fuente de audio.",
      "Definir puerto de salida.",
      "Presionar Iniciar para levantar el servidor.",
      "Abrir la URL local para ver subtitulos en vivo.",
      "Guardar la transcripcion si se requiere.",
    ],
  },
  gaussian: {
    title: "Gaussian Splatting",
    summary:
      "Pipeline para crear escenas 3D con gaussian splats a partir de imagenes o video.",
    what_is:
      "Herramienta de reconstruccion 3D que genera dataset, entrena splats y produce un visor interactivo.",
    purpose:
      "Convertir capturas reales en escenas navegables para docencia, demos o prototipos.",
    use_cases: [
      "Digitalizacion de objetos o espacios.",
      "Previsualizacion 3D rapida para proyectos.",
      "Material didactico para vision por computadora.",
      "Demos de laboratorio en 3D.",
      "Comparativa de calidad entre datasets.",
    ],
    how_to: [
      "Preparar un set de imagenes o video.",
      "Cargar la ruta de entrada en el modulo.",
      "Configurar resolucion y parametros de entrenamiento.",
      "Ejecutar el proceso y monitorear el avance.",
      "Abrir el visor para navegar la escena.",
      "Exportar resultados para uso externo si aplica.",
    ],
  },
  ml_sharp: {
    title: "ML-SHARP",
    summary:
      "Reconstruccion rapida con gaussian splats enfocada en velocidad y previsualizacion.",
    what_is:
      "Implementacion optimizada para pruebas rapidas con presets de calidad y rendimiento.",
    purpose:
      "Obtener resultados tempranos para evaluar captura antes de un pipeline completo.",
    use_cases: [
      "Validar la calidad de un set de imagenes.",
      "Comparar configuraciones de render.",
      "Prototipar escenas 3D en poco tiempo.",
      "Demostraciones en clase con tiempos reducidos.",
    ],
    how_to: [
      "Instalar dependencias del backend.",
      "Seleccionar carpeta de entrada y salida.",
      "Elegir preset de calidad (rapido/medio/alto).",
      "Configurar dispositivo (CPU/GPU).",
      "Ejecutar y abrir la previsualizacion.",
    ],
  },
  model_3d: {
    title: "Generacion de modelo 3D",
    summary:
      "Orquestador para crear modelos 3D desde imagenes o video con distintos backends.",
    what_is:
      "Hub que integra Hunyuan, Reconv u otros pipelines 3D en una misma vista.",
    purpose:
      "Simplificar el acceso a varios backends 3D y comparar resultados.",
    use_cases: [
      "Prototipos rapidos de objetos.",
      "Generacion de assets para cursos.",
      "Comparativas de backends 3D.",
      "Modelos para visualizacion de proyectos.",
    ],
    how_to: [
      "Seleccionar el backend preferido.",
      "Instalar backend y pesos si no estan.",
      "Cargar imagenes o video de entrada.",
      "Ajustar parametros de calidad y resolucion.",
      "Ejecutar y abrir la salida 3D.",
    ],
  },
  spotedit: {
    title: "SpotEdit",
    summary:
      "Edicion localizada con mascara para corregir regiones especificas en imagenes.",
    what_is:
      "Modulo de inpainting/outpainting con modelos Diffusion y prompts guiados.",
    purpose:
      "Modificar zonas puntuales sin rehacer toda la imagen.",
    use_cases: [
      "Eliminar objetos no deseados.",
      "Reemplazar partes con instrucciones textuales.",
      "Correcciones finas en material visual.",
      "Probar variantes de una misma imagen.",
    ],
    how_to: [
      "Instalar backend y dependencias.",
      "Cargar imagen de entrada.",
      "Definir o subir la mascara.",
      "Escribir prompt y parametros.",
      "Ejecutar y revisar el resultado.",
      "Iterar hasta obtener la version final.",
    ],
  },
  hy_motion: {
    title: "HY-Motion 1.0",
    summary:
      "Generacion y edicion de movimiento para personajes u objetos a partir de texto.",
    what_is:
      "Modelo Hunyuan para animacion y motion editing con presets de duracion.",
    purpose:
      "Producir secuencias de movimiento para demos, clases o prototipos.",
    use_cases: [
      "Animaciones cortas para presentaciones.",
      "Demos de movimiento en cursos.",
      "Pruebas de prompts de movimiento.",
      "Generacion de variantes de una misma accion.",
    ],
    how_to: [
      "Instalar backend y dependencias.",
      "Descargar pesos si aplica.",
      "Definir prompt de movimiento.",
      "Ajustar duracion y calidad.",
      "Ejecutar y abrir la salida.",
    ],
  },
  proedit: {
    title: "ProEdit",
    summary:
      "Edicion avanzada con inversion para mantener consistencia en imagen y video.",
    what_is:
      "Workflow de edicion con inversion y control fino de atributos.",
    purpose:
      "Lograr cambios precisos sin perder identidad visual.",
    use_cases: [
      "Edicion profesional de imagen.",
      "Ajustes de estilo manteniendo estructura.",
      "Correccion de video con consistencia temporal.",
      "Pruebas comparativas de edicion avanzada.",
    ],
    how_to: [
      "Instalar el backend requerido.",
      "Cargar insumos (imagen o secuencia de video).",
      "Configurar parametros de inversion.",
      "Ejecutar el pipeline de edicion.",
      "Revisar salida y ajustar parametros si es necesario.",
    ],
  },
  neutts: {
    title: "NeuTTS",
    summary:
      "Texto a voz local con variantes de calidad y perfiles de voz.",
    what_is:
      "Motor TTS con modelos Neuphonic (Air/Nano) y control de salida.",
    purpose:
      "Generar audio local para narraciones sin depender de servicios externos.",
    use_cases: [
      "Narracion de contenidos para clases.",
      "Generacion de voz para demos.",
      "Lectura automatica de textos largos.",
      "Pruebas de voces en materiales didacticos.",
    ],
    how_to: [
      "Seleccionar modelo y variante de voz.",
      "Instalar dependencias si se solicita.",
      "Cargar texto o archivo de entrada.",
      "Ajustar velocidad y tono si aplica.",
      "Generar audio y descargar la salida.",
    ],
  },
  finetune_glm: {
    title: "Fine-tune GLM-4.7",
    summary:
      "Asistente guiado para entrenar GLM-4.7-Flash con Unsloth y exportar a GGUF.",
    what_is:
      "Pipeline local LoRA: prepara dataset, entrena, evalua y exporta el modelo.",
    purpose:
      "Personalizar GLM-4.7-Flash con conocimiento institucional o de dominio.",
    use_cases: [
      "Entrenar respuestas institucionales.",
      "Ajustar un modelo para una materia.",
      "Adaptar un tono o estilo de redaccion.",
      "Crear un asistente para un area especifica.",
      "Prototipar un modelo con datos propios.",
    ],
    how_to: [
      "Instalar dependencias desde el modulo.",
      "Preparar dataset JSONL (messages o conversations).",
      "Validar que el dataset tenga prompts y respuestas.",
      "Definir hiperparametros (epochs, batch, lr, rank).",
      "Iniciar el entrenamiento y seguir el log.",
      "Exportar a GGUF al finalizar.",
      "Mover el GGUF a la carpeta de modelos y probarlo en Chat AI.",
    ],
  },
  research_assistant: {
    title: "Asistente de Investigacion",
    summary:
      "Biblioteca local de PDFs con indexado, resumenes y consultas RAG offline.",
    what_is:
      "Gestor de documentos con busqueda semantica y respuestas asistidas.",
    purpose:
      "Organizar material academico y responder preguntas sin salir del entorno.",
    use_cases: [
      "Resumen de papers o informes.",
      "Busqueda de citas y referencias.",
      "Generar bibliografia APA/IEEE.",
      "Responder preguntas sobre multiples documentos.",
    ],
    how_to: [
      "Agregar PDFs a la biblioteca.",
      "Generar indice para habilitar busqueda.",
      "Esperar el procesamiento inicial.",
      "Consultar por tema o pregunta.",
      "Exportar resumenes o citas cuando aplique.",
    ],
  },
  klein: {
    title: "Flux 2 Klein",
    summary:
      "Generacion de imagenes texto a imagen con presets de calidad y resolucion.",
    what_is:
      "Modelo local para producir imagenes con prompts, estilos y seeds.",
    purpose:
      "Crear material visual para clases, presentaciones o prototipos.",
    use_cases: [
      "Material visual para clases.",
      "Prototipos rapidos de imagenes.",
      "Exploracion de estilos artisticos.",
      "Borradores visuales para proyectos.",
    ],
    how_to: [
      "Instalar dependencias del backend.",
      "Definir prompt y parametros (resolucion, steps, seed).",
      "Ejecutar la generacion.",
      "Revisar y guardar la salida.",
      "Iterar variaciones si es necesario.",
    ],
  },
  hyworld: {
    title: "HunyuanWorld-Mirror",
    summary:
      "Reconstruccion 3D con HunyuanWorld para escenas y objetos.",
    what_is:
      "Backend de reconstruccion 3D con flujo guiado y visor de resultados.",
    purpose:
      "Transformar imagenes en escenas 3D para exploracion o demostraciones.",
    use_cases: [
      "Demos de vision 3D en cursos.",
      "Reconstruccion rapida de objetos.",
      "Evaluacion de datasets de captura.",
      "Visualizacion de escenas para presentaciones.",
    ],
    how_to: [
      "Instalar backend y dependencias.",
      "Descargar pesos requeridos.",
      "Cargar imagenes de entrada.",
      "Ejecutar el proceso de reconstruccion.",
      "Abrir el visor y exportar si aplica.",
    ],
  },
  cyberscraper: {
    title: "CyberScraper 2077",
    summary:
      "Scraping asistido con UI, plantillas y soporte LLM (OpenAI/Gemini/Ollama).",
    what_is:
      "Interfaz para recolectar datos web, limpiar resultados y exportar datasets.",
    purpose:
      "Recolectar informacion estructurada con ayuda de modelos y flujos guiados.",
    use_cases: [
      "Recoleccion de datos academicos.",
      "Scraping de catalogos institucionales.",
      "Normalizacion de textos para analisis.",
      "Generacion de datasets para investigacion.",
    ],
    how_to: [
      "Instalar backend y dependencias.",
      "Configurar llaves API si aplica.",
      "Definir URL y plantilla de scraping.",
      "Ejecutar la captura y revisar resultados.",
      "Exportar a CSV/JSON segun necesidad.",
    ],
  },
};
