use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    Manager,
};
use tauri_plugin_autostart::ManagerExt;

#[cfg(target_os = "macos")]
static APP_SHUTTING_DOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(target_os = "macos")]
use mac_notification_sys::{MainButton, Notification, NotificationResponse};

#[cfg(target_os = "macos")]
#[derive(serde::Deserialize)]
struct ApprovalList { items: Vec<ApprovalItem> }

#[cfg(target_os = "macos")]
#[derive(serde::Deserialize, Clone)]
struct ApprovalItem { kind: String, id: String, title: String, body: String }

#[cfg(target_os = "macos")]
fn start_approval_notifications() {
    std::thread::spawn(|| {
        let _ = mac_notification_sys::set_application("ai.family.desktop");
        let mut shown = std::collections::HashSet::<String>::new();
        loop {
            let output = std::process::Command::new("/usr/bin/curl")
                .args(["-fsS", "http://127.0.0.1:8000/api/approvals?member_id=*"])
                .output();
            if let Ok(output) = output {
                if let Ok(list) = serde_json::from_slice::<ApprovalList>(&output.stdout) {
                    let pending: std::collections::HashSet<String> = list.items.iter().map(|x| format!("{}:{}", x.kind, x.id)).collect();
                    shown.retain(|key| pending.contains(key));
                    for item in list.items {
                        let key = format!("{}:{}", item.kind, item.id);
                        if shown.insert(key) {
                            std::thread::spawn(move || {
                                let response = Notification::new()
                                    .title(&item.title)
                                    .subtitle("Family AI Approval Center")
                                    .message(&item.body)
                                    .main_button(MainButton::SingleAction("Approve"))
                                    .close_button("Deny")
                                    .default_sound()
                                    .wait_for_click(true)
                                    .send();
                                let decision = match response {
                                    Ok(NotificationResponse::ActionButton(_)) => Some("approve"),
                                    Ok(NotificationResponse::CloseButton(_)) => Some("deny"),
                                    _ => None,
                                };
                                if let Some(decision) = decision {
                                    if APP_SHUTTING_DOWN.load(std::sync::atomic::Ordering::SeqCst) {
                                        return;
                                    }
                                    let url = format!("http://127.0.0.1:8000/api/approvals/{}/{}/{}?member_id=*", item.kind, item.id, decision);
                                    let _ = std::process::Command::new("/usr/bin/curl").args(["-fsS", "-X", "POST", &url]).output();
                                }
                            });
                        }
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_secs(5));
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .setup(|app| {
            let _ = app.autolaunch().enable();
            #[cfg(target_os = "macos")]
            start_approval_notifications();
            let show = MenuItem::with_id(app, "show", "Open Family AI", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit Family AI", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &quit])?;
            TrayIconBuilder::new()
                .icon(app.default_window_icon().cloned().unwrap())
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building Family AI")
        .run(|app, event| {
            #[cfg(target_os = "macos")]
            match event {
                tauri::RunEvent::Reopen { .. } => {
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.show();
                        let _ = window.unminimize();
                        let _ = window.set_focus();
                    }
                }
                tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit => {
                    APP_SHUTTING_DOWN.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                _ => {}
            }
        });
}
