# Angular

Angular divide una aplicación en piezas claras:
módulos, componentes, servicios y rutas.
Todo vive dentro de `src/`.
`main.ts` arranca la app, `app.module.ts` define el módulo raíz, `app.component.*` es el primer
componente que se muestra y `index.html` es el archivo HTML principal.
Las dependencias externas se guardan en `package.json` y se instalan con npm.
La configuración interna de construcción y ejecución está en `angular.json`.

Una app Angular se organiza así:
los **módulos** agrupan funcionalidad; los **componentes** manejan vista y lógica local; los
**servicios** guardan lógica compartida o acceso a datos; y el **router** construye la
navegación.
El data binding y los lifecycle hooks controlan comunicación y ciclos de vida.

Comandos esenciales del CLI:
`ng new name` crea un proyecto.
`ng serve` ejecuta el servidor de desarrollo.
`ng build` genera la versión compilada.
`ng test` corre pruebas.
`ng lint` analiza el código.
`ng generate` o `ng g` crea nuevas piezas del proyecto.

Comandos para generar partes específicas:
`ng g module name` crea un módulo.
`ng g module name --routing` crea módulo con archivo de rutas.
`ng g component name` crea un componente.
`ng g service name` crea un servicio.
`ng g pipe name` crea un pipe.
`ng g directive name` crea una directiva.
`ng g guard name` crea un guard.
`ng g interface name` crea una interfaz TypeScript.

Para estudiar de forma enfocada:
estructura de carpetas, función de cada archivo principal, decorators más usados, flujo de
datos, binding, lifecycle hooks, sistema de módulos y router, y manejo básico del CLI.

Un **módulo** (NgModule) agrupa partes de la app:
componentes, pipes, servicios.
Define qué cosas pertenecen a ese módulo y qué cosas exporta.
Cada app tiene al menos `AppModule`.

Un **componente** es una combinación de tres partes:
una plantilla HTML, un archivo TypeScript con lógica y un archivo de estilos opcional.
El componente no tiene un “spec.ts” para lógica; ese archivo es solo para tests unitarios.
La lógica del componente está en el `.ts`.

Un componente típico:
`component.ts` lógica, propiedades, métodos, bindings `component.html` HTML de la vista
`component.scss|css` estilos del componente `component.spec.ts` solo para tests, no para
lógica

El HTML y CSS son de ese componente, no de todo el módulo.
Un módulo puede tener muchos componentes.

Las **rutas** no van en cada módulo por defecto.
Hay dos enfoques:

1. Una app pequeña tiene un solo archivo de rutas:
   `app-routing.module.ts`.
2. Una app grande divide rutas por módulos (lazy loading).
   Cada módulo puede tener su propio `module-routing.ts`, pero solo si eliges esa estructura.

Cómo funciona realmente el routing:
El `AppRoutingModule` define URLs y qué componente cargar.
Si un módulo se carga en lazy-loading, entonces ese módulo tiene un archivo de rutas propio,
pero esto no es obligatorio.

Resumiendo bien el modelo:

Un **módulo** contiene:
• componentes • servicios • pipes • imports/exports • opcionalmente un routing propio (solo si
quieres lazy loading)

Un **componente** contiene:
• `.ts` con la lógica • `.html` con la vista • `.scss`/`.css` con estilos • `.spec.ts` solo
para tests

El **routing** está centralizado en `app-routing.module.ts` o distribuido por módulos si usas
lazy loading.

# Flutter

Flutter organiza una app alrededor de widgets.
Todo es un widget.
El proyecto se crea vacío con una estructura estándar y se compila para Android, iOS, web y
desktop.
El archivo central es `lib/main.dart`, que ejecuta `runApp()` y carga tu widget raíz,
normalmente una clase `MyApp`.
Dentro usas `MaterialApp` o `CupertinoApp`, defines rutas, temas y el árbol de widgets.

Para iniciar un proyecto, se usa `flutter create name`.
Para correrlo, `flutter run`.
Para construir un release, `flutter build apk`, `flutter build ios`, o el tipo correspondiente.
La depuración se hace con `flutter doctor` para revisar instalación y `flutter devices` para
ver dispositivos conectados.

Las dependencias externas se definen en `pubspec.yaml`.
Allí se agregan paquetes en la sección `dependencies:` y assets en la sección `assets:`.
Se instalan con `flutter pub get`.
Los assets como imágenes o fuentes se guardan normalmente en `assets/`.
Flutter necesita que declares estos recursos en `pubspec.yaml` para incluirlos en la build.

El flujo básico para agregar código es crear nuevos widgets en `lib/`.
Un proyecto típico tiene carpetas como `lib/screens/`, `lib/widgets/` y `lib/services/` según
la organización que elijas.
Los archivos más relevantes son `main.dart` para el arranque, `pubspec.yaml` para dependencias
y `android/` e `ios/` para ajustes específicos de cada plataforma.
Si necesitas navegar entre pantallas, usas `Navigator` o un sistema de rutas.
Para administrar estado existen varias opciones como `Provider`, `Riverpod` o `Bloc`, cada una
agregada como dependencia.

Comandos más usados:
`flutter create name` crea un proyecto.
`flutter run` ejecuta en dispositivo o emulador.
`flutter build` genera builds para cada plataforma.
`flutter pub get` instala dependencias.
`flutter pub add package` agrega una dependencia desde CLI.
`flutter analyze` revisa el código.
`flutter test` corre pruebas.
