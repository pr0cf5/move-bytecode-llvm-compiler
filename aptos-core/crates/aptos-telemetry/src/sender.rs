// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use crate::metrics::{self, increment_log_ingest_failures_by, increment_log_ingest_successes_by};
use anyhow::{anyhow, Error};
use aptos_config::config::{NodeConfig, RoleType};
use aptos_crypto::{
    noise::{self, NoiseConfig},
    x25519,
};
use aptos_infallible::{Mutex, RwLock};
use aptos_logger::debug;
use aptos_telemetry_service::types::{
    auth::{AuthRequest, AuthResponse},
    response::IndexResponse,
    telemetry::TelemetryDump,
};
use aptos_types::{chain_id::ChainId, PeerId};
use flate2::{write::GzEncoder, Compression};
use prometheus::{default_registry, Registry};
use reqwest::{header::CONTENT_ENCODING, RequestBuilder, Response, StatusCode};
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};
use std::{io::Write, sync::Arc};

struct AuthContext {
    noise_config: Option<NoiseConfig>,
    token: RwLock<Option<String>>,
    server_public_key: Mutex<Option<x25519::PublicKey>>,
}

impl AuthContext {
    fn new(node_config: &NodeConfig) -> Self {
        Self {
            noise_config: node_config.identity_key().map(NoiseConfig::new),
            token: RwLock::new(None),
            server_public_key: Mutex::new(None),
        }
    }
}

#[derive(Clone)]
pub(crate) struct TelemetrySender {
    base_url: String,
    chain_id: ChainId,
    peer_id: PeerId,
    role_type: RoleType,
    inner_client: reqwest::Client,
    client: ClientWithMiddleware,
    auth_context: Arc<AuthContext>,
}

impl TelemetrySender {
    pub fn new(base_url: String, chain_id: ChainId, node_config: &NodeConfig) -> Self {
        let retry_policy = ExponentialBackoff::builder().build_with_max_retries(3);
        // This is a workaround to enable us to clone RequestBuilder for retries.
        // reqwest_middleware::RequestBuilder's try_clone moves self and returns a new Self, which defeats
        // the purpose of cloning. We will use the reqwest::Client to create a reqwest::RequestBuilder,
        // which is clonable and use the ClientWithMiddleware to actually send requests.
        let inner_client = reqwest::Client::new();
        let client = ClientBuilder::new(inner_client.clone())
            .with(RetryTransientMiddleware::new_with_policy(retry_policy))
            .build();
        Self {
            base_url,
            chain_id,
            peer_id: node_config.peer_id().unwrap_or(PeerId::ZERO),
            role_type: node_config.base.role,
            inner_client,
            client,
            auth_context: Arc::new(AuthContext::new(node_config)),
        }
    }

    // sends an authenticated request to the telemetry service, automatically adding an auth token
    // This function does not work with streaming bodies at the moment and will panic if you try so.
    pub async fn send_authenticated_request(
        &self,
        request_builder: RequestBuilder,
    ) -> Result<Response, anyhow::Error> {
        let token = self.get_auth_token().await?;

        let request = request_builder
            .try_clone()
            .expect("Could not clone request_builder")
            .bearer_auth(token)
            .build()?;

        let mut response = self.client.execute(request).await?;

        // do 1 retry if the first attempt failed
        if response.status() == StatusCode::UNAUTHORIZED {
            // looks like request failed due to auth error. Let's get a new a fresh token. If this fails again we'll just return the error.
            self.reset_token();
            let token = self.get_auth_token().await?;
            let request = request_builder.bearer_auth(token).build()?;
            response = self.client.execute(request).await?;
        }
        Ok(response)
    }

    pub(crate) async fn push_prometheus_metrics(
        &self,
        registry: &Registry,
    ) -> Result<(), anyhow::Error> {
        debug!("Sending Prometheus Metrics");

        let scraped_metrics =
            prometheus::TextEncoder::new().encode_to_string(&registry.gather())?;

        let mut gzip_encoder = GzEncoder::new(Vec::new(), Compression::default());
        gzip_encoder.write_all(scraped_metrics.as_bytes())?;
        let compressed_bytes = gzip_encoder.finish()?;

        let response = self
            .send_authenticated_request(
                self.inner_client
                    .post(format!("{}/api/v1/ingest/metrics", self.base_url))
                    .header(CONTENT_ENCODING, "gzip")
                    .body(compressed_bytes),
            )
            .await;

        match response {
            Err(e) => Err(anyhow!("Prometheus Metrics push failed: {}", e)),
            Ok(response) => {
                if response.status().is_success() {
                    Ok(())
                } else {
                    Err(anyhow!(
                        "Prometheus Metrics push failed with response: {}, body: {}",
                        response.status(),
                        response
                            .text()
                            .await
                            .unwrap_or_else(|_| "empty body".to_string()),
                    ))
                }
            },
        }
    }

    pub(crate) async fn try_push_prometheus_metrics(&self) {
        self.push_prometheus_metrics(default_registry())
            .await
            .map_or_else(
                |e| debug!("Failed to push Prometheus Metrics: {}", e),
                |_| debug!("Prometheus Metrics pushed successfully."),
            );
    }

    pub async fn try_send_logs(&self, batch: Vec<String>) {
        if let Ok(json) = serde_json::to_string(&batch) {
            let len = json.len();

            match self.post_logs(json.as_bytes()).await {
                Ok(_) => {
                    increment_log_ingest_successes_by(batch.len() as u64);
                    debug!("Sent log of length: {}", len);
                },
                Err(error) => {
                    increment_log_ingest_failures_by(batch.len() as u64);
                    debug!("Failed send log of length: {} with error: {}", len, error);
                },
            }
        } else {
            debug!("Failed json serde of batch: {:?}", batch);
        }
    }

    async fn post_logs(&self, json: &[u8]) -> Result<Response, anyhow::Error> {
        debug!("Sending logs");

        let mut gzip_encoder = GzEncoder::new(Vec::new(), Compression::default());
        gzip_encoder.write_all(json)?;
        let compressed_bytes = gzip_encoder.finish()?;

        // Send the request and wait for a response
        let response = self
            .send_authenticated_request(
                self.inner_client
                    .post(format!("{}/api/v1/ingest/logs", self.base_url))
                    .header(CONTENT_ENCODING, "gzip")
                    .body(compressed_bytes),
            )
            .await?;

        // Process the result
        error_for_status_with_body(response).await
    }

    pub async fn try_send_custom_metrics(&self, event_name: String, telemetry_dump: TelemetryDump) {
        match self.post_custom_metrics(&telemetry_dump.clone()).await {
            Ok(_) => {
                metrics::increment_telemetry_service_successes(&event_name);
                debug!("Custom metrics with name {} sent successfully.", event_name);
            },
            Err(e) => {
                metrics::increment_telemetry_service_failures(&event_name);
                debug!("Failed to send custom metrics: {}", e);
            },
        }
    }

    async fn post_custom_metrics(
        &self,
        telemetry_dump: &TelemetryDump,
    ) -> Result<Response, anyhow::Error> {
        // Send the request and wait for a response
        let response = self
            .send_authenticated_request(
                self.inner_client
                    .post(format!("{}/api/v1/ingest/custom-event", self.base_url))
                    .json::<TelemetryDump>(telemetry_dump),
            )
            .await?;

        error_for_status_with_body(response).await
    }

    async fn get_auth_token(&self) -> Result<String, Error> {
        // Try to read the token holding a read lock
        let token = { self.auth_context.token.read().as_ref().cloned() };
        match token {
            Some(token) => Ok(token),
            None => {
                let token = self.authenticate().await?;
                *self.auth_context.token.write() = Some(token.clone());
                Ok(token)
            },
        }
    }

    async fn get_public_key_from_server(&self) -> Result<x25519::PublicKey, anyhow::Error> {
        let response = self
            .client
            .get(format!("{}/api/v1/", self.base_url))
            .send()
            .await?;

        match error_for_status_with_body(response).await {
            Ok(response) => {
                let response_payload = response.json::<IndexResponse>().await?;
                Ok(response_payload.public_key)
            },
            Err(err) => Err(anyhow!("Error getting server public key. {}", err)),
        }
    }

    async fn server_public_key(&self) -> Result<x25519::PublicKey, anyhow::Error> {
        let server_public_key = { *self.auth_context.server_public_key.lock() };
        match server_public_key {
            Some(key) => Ok(key),
            None => {
                let public_key = self.get_public_key_from_server().await?;
                *self.auth_context.server_public_key.lock() = Some(public_key);
                Ok(public_key)
            },
        }
    }

    fn reset_token(&self) {
        *self.auth_context.token.write() = None;
        *self.auth_context.server_public_key.lock() = None;
    }

    pub async fn authenticate(&self) -> Result<String, anyhow::Error> {
        let noise_config = match &self.auth_context.noise_config {
            Some(config) => config,
            None => return Err(anyhow!("Cannot send telemetry without private key")),
        };
        let server_public_key = self.server_public_key().await?;

        // buffer to first noise handshake message
        let mut client_noise_msg = vec![0; noise::handshake_init_msg_len(0)];

        // build the prologue (chain_id | peer_id | server_public_key)
        const CHAIN_ID_LENGTH: usize = 1;
        const ID_SIZE: usize = CHAIN_ID_LENGTH + PeerId::LENGTH;
        const PROLOGUE_SIZE: usize = CHAIN_ID_LENGTH + PeerId::LENGTH + x25519::PUBLIC_KEY_SIZE;
        let mut prologue = [0; PROLOGUE_SIZE];
        prologue[..CHAIN_ID_LENGTH].copy_from_slice(&[self.chain_id.id()]);
        prologue[CHAIN_ID_LENGTH..ID_SIZE].copy_from_slice(self.peer_id.as_ref());
        prologue[ID_SIZE..PROLOGUE_SIZE].copy_from_slice(server_public_key.as_slice());

        let mut rng = rand::rngs::OsRng;

        // craft first handshake message  (-> e, es, s, ss)
        let initiator_state = noise_config
            .initiate_connection(
                &mut rng,
                &prologue,
                server_public_key,
                None,
                &mut client_noise_msg,
            )
            .unwrap();

        let auth_request = AuthRequest {
            chain_id: self.chain_id,
            peer_id: self.peer_id,
            role_type: self.role_type,
            server_public_key,
            handshake_msg: client_noise_msg,
        };

        let response = self
            .client
            .post(format!("{}/api/v1/auth", self.base_url))
            .json::<AuthRequest>(&auth_request)
            .send()
            .await?;

        let resp = match error_for_status_with_body(response).await {
            Ok(response) => Ok(response.json::<AuthResponse>().await?),
            Err(err) => {
                debug!(
                    "[telemetry-client] Error sending authentication request: {}",
                    err,
                );
                Err(anyhow!("error {}", err))
            },
        }?;

        let (response_payload, _) = noise_config
            .finalize_connection(initiator_state, resp.handshake_msg.as_slice())
            .unwrap();

        let jwt = String::from_utf8(response_payload)?;

        Ok(jwt)
    }

    pub(crate) async fn check_chain_access(&self, chain_id: ChainId) -> bool {
        debug!("checking chain access for chain id {}", chain_id);
        let response = self
            .client
            .get(format!(
                "{}/api/v1/chain-access/{}",
                self.base_url, chain_id
            ))
            .send()
            .await;

        match response {
            Ok(response) => match error_for_status_with_body(response).await {
                Ok(response) => response.json::<bool>().await.unwrap_or(true),
                Err(e) => {
                    debug!("Unable to check chain access {}", e);
                    true
                },
            },
            Err(e) => {
                debug!("Unable to check chain access {}", e);
                true
            },
        }
    }

    pub(crate) async fn get_telemetry_log_env(&self) -> Option<String> {
        let response = self
            .send_authenticated_request(
                self.inner_client
                    .get(format!("{}/api/v1/config/env/telemetry-log", self.base_url)),
            )
            .await;

        match response {
            Ok(response) => match error_for_status_with_body(response).await {
                Ok(response) => response.json::<Option<String>>().await.unwrap_or_default(),
                Err(e) => {
                    debug!("Unable to get telemetry log env: {}", e);
                    None
                },
            },
            Err(e) => {
                debug!("Unable to check chain access {}", e);
                None
            },
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::metrics::{APTOS_TELEMETRY_SERVICE_FAILURE, APTOS_TELEMETRY_SERVICE_SUCCESS};
    use aptos_crypto::Uniform;
    use aptos_telemetry_service::types::telemetry::TelemetryEvent;
    use httpmock::MockServer;
    use prometheus::{register_int_counter_vec_with_registry, Registry};
    use std::{
        collections::BTreeMap,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[tokio::test]
    async fn test_server_public_key() {
        let mut rng = rand::thread_rng();
        let private_key = x25519::PrivateKey::generate(&mut rng);

        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET").path("/api/v1/");
            then.status(200).json_body_obj(&IndexResponse {
                public_key: private_key.public_key(),
            });
        });

        let node_config = NodeConfig::default();
        let client = TelemetrySender::new(server.base_url(), ChainId::default(), &node_config);

        let result1 = client.server_public_key().await;
        let result2 = client.server_public_key().await;

        // Should call the server once and cache the key
        assert_eq!(mock.hits(), 1);
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap(), private_key.public_key());
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap(), private_key.public_key());

        client.reset_token();

        let result3 = client.server_public_key().await;
        assert_eq!(mock.hits(), 2);
        assert!(result3.is_ok());
        assert_eq!(result3.unwrap(), private_key.public_key());
    }

    #[tokio::test]
    async fn test_post_custom_metrics() {
        let mut telemetry_event = TelemetryEvent {
            name: "sample-event".into(),
            params: BTreeMap::new(),
        };
        telemetry_event
            .params
            .insert("key-1".into(), "value-1".into());
        let telemetry_dump = TelemetryDump {
            client_id: "client-1".into(),
            user_id: "user-1".into(),
            timestamp_micros: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_micros()
                .to_string(),
            events: vec![],
        };

        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("POST")
                .header("Authorization", "Bearer SECRET_JWT_TOKEN")
                .path("/api/v1/ingest/custom-event")
                .json_body_obj(&telemetry_dump);
            then.status(200);
        });

        let node_config = NodeConfig::default();
        let client = TelemetrySender::new(server.base_url(), ChainId::default(), &node_config);
        {
            *client.auth_context.token.write() = Some("SECRET_JWT_TOKEN".into());
        }

        let result = client.post_custom_metrics(&telemetry_dump).await;

        mock.assert();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_try_send_metrics_retry_unauthorized() {
        let event_name = "sample-event";
        let mut telemetry_event = TelemetryEvent {
            name: event_name.into(),
            params: BTreeMap::new(),
        };
        telemetry_event
            .params
            .insert("key-1".into(), "value-1".into());
        let telemetry_dump = TelemetryDump {
            client_id: "client-1".into(),
            user_id: "user-1".into(),
            timestamp_micros: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_micros()
                .to_string(),
            events: vec![],
        };

        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("POST").path("/api/v1/ingest/custom-event");
            then.status(401);
        });

        let node_config = NodeConfig::default();
        let client = TelemetrySender::new(server.base_url(), ChainId::default(), &node_config);
        {
            *client.auth_context.token.write() = Some("SECRET_JWT_TOKEN".into());
        }

        client
            .try_send_custom_metrics(event_name.into(), telemetry_dump)
            .await;

        mock.assert_hits(1);
        assert_eq!(
            APTOS_TELEMETRY_SERVICE_SUCCESS
                .with_label_values(&[event_name])
                .get(),
            0
        );
        assert_eq!(
            APTOS_TELEMETRY_SERVICE_FAILURE
                .with_label_values(&[event_name])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn test_push_prometheus_metrics() {
        // Initialize a local prometheus registry
        // Using the global registry will conflict will other tests that increment counters
        let test_registry = Registry::default();

        let counter = register_int_counter_vec_with_registry!(
            "aptos_telemetry_service_success",
            "Number of telemetry events successfully sent to telemetry service",
            &["event_name"],
            test_registry
        )
        .unwrap();

        counter.with_label_values(&["test-event"]).inc();

        let scraped_metrics = prometheus::TextEncoder::new()
            .encode_to_string(&test_registry.gather())
            .unwrap();

        let mut gzip_encoder = GzEncoder::new(Vec::new(), Compression::default());
        gzip_encoder.write_all(scraped_metrics.as_bytes()).unwrap();
        let expected_compressed_bytes = gzip_encoder.finish().unwrap();

        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("POST")
                .header("Authorization", "Bearer SECRET_JWT_TOKEN")
                .path("/api/v1/ingest/metrics")
                .body(String::from_utf8_lossy(&expected_compressed_bytes));
            then.status(200);
        });

        let node_config = NodeConfig::default();
        let client = TelemetrySender::new(server.base_url(), ChainId::default(), &node_config);
        {
            *client.auth_context.token.write() = Some("SECRET_JWT_TOKEN".into());
        }

        let result = client.push_prometheus_metrics(&test_registry).await;

        mock.assert();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_post_logs() {
        let batch = vec!["log1".to_string(), "log2".to_string()];
        let json = serde_json::to_string(&batch);
        assert!(json.is_ok());

        let mut gzip_encoder = GzEncoder::new(Vec::new(), Compression::default());
        gzip_encoder.write_all(json.unwrap().as_bytes()).unwrap();
        let expected_compressed_bytes = gzip_encoder.finish().unwrap();

        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("POST")
                .header("Authorization", "Bearer SECRET_JWT_TOKEN")
                .path("/api/v1/ingest/logs")
                .body(String::from_utf8_lossy(&expected_compressed_bytes));
            then.status(200);
        });

        let node_config = NodeConfig::default();
        let client = TelemetrySender::new(server.base_url(), ChainId::default(), &node_config);
        {
            *client.auth_context.token.write() = Some("SECRET_JWT_TOKEN".into());
        }

        client.try_send_logs(batch).await;

        mock.assert();
    }

    #[tokio::test]
    async fn test_check_chain_access() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET").path("/api/v1/chain-access/24");
            then.status(200).json_body(true);
        });

        let client = TelemetrySender::new(
            server.base_url(),
            ChainId::default(),
            &NodeConfig::default(),
        );
        assert!(client.check_chain_access(ChainId::new(24)).await);

        mock.assert();
    }
}

async fn error_for_status_with_body(response: Response) -> Result<Response, anyhow::Error> {
    if response.status().is_client_error() || response.status().is_server_error() {
        Err(anyhow!(
            "HTTP status error ({}) for url ({}): {}",
            response.status(),
            response.url().clone(),
            response.text().await?,
        ))
    } else {
        Ok(response)
    }
}
