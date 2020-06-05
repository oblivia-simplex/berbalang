use std::io::Write;

/// This function initializes the Berbalang logger.
///
pub fn init(population_name: &str) {
    let population_name = population_name.to_owned();
    env_logger::from_env(env_logger::Env::from("BERBALANG_LOG"))
        .format(move |f, record| {
            use env_logger::fmt::Color;
            use log::Level::*;

            let path = match record.level() {
                Error | Debug | Warn | Trace => format!(
                    ":{}:{}",
                    record.file().unwrap_or(""),
                    record.line().unwrap_or(0),
                ),
                Info => "".to_string(),
            };

            let mut time_style = f.style();
            time_style
                .set_color(Color::Rgb(100, 100, 100))
                .set_bold(false);
            let time = time_style.value(format!("[{}]", f.timestamp()));

            let mut level_style = f.style();
            let color = match record.level() {
                Error => Color::Red,
                Warn => Color::Yellow,
                Info => Color::Green,
                Trace => Color::Magenta,
                Debug => Color::Cyan,
            };
            level_style.set_color(color).set_bold(true);
            let level = level_style.value(record.level());

            let mut pop_style = f.style();
            pop_style.set_color(Color::White).set_bold(true);
            let population_name = pop_style.value(&population_name);

            writeln!(
                f,
                "[{level}{path}]{time} {pop_name} => {record:#x?}",
                time = time,
                path = path,
                level = level,
                pop_name = population_name,
                record = record.args(),
            )
        })
        .init();
}
